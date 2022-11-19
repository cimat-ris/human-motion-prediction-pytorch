
"""Sequence-to-sequence model for human motion prediction."""
import random
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MotionPredictor(nn.Module):
	"""Sequence-to-sequence model for human motion prediction"""
	def __init__(self,source_seq_len,target_seq_len,
		rnn_size, # recurrent layer hidden size
		batch_size,learning_rate,learning_rate_decay_factor,
		number_of_actions,dropout=0.3):

		"""Args:
		source_seq_len: length of the input sequence.
		target_seq_len: length of the target sequence.
		rnn_size: number of units in the rnn.
		batch_size: the size of the batches used during training;
			the model construction is independent of batch_size, so it can be
			changed after initialization if this is convenient, e.g., for decoding.
		learning_rate: learning rate to start with.
		learning_rate_decay_factor: decay learning rate by this much when needed.
		number_of_actions: number of classes we have.
		"""
		super(MotionPredictor, self).__init__()

		self.human_dofs     = 54
		self.input_size     = self.human_dofs + number_of_actions

		logging.info("Input size is {}".format(self.input_size))
		self.source_seq_len = source_seq_len
		self.target_seq_len = target_seq_len
		self.rnn_size       = rnn_size
		self.batch_size     = batch_size
		self.dropout        = dropout

		# === Create the RNN that will summarizes the state ===
		self.cell           = torch.nn.GRUCell(self.input_size, self.rnn_size)
		self.fc1            = nn.Linear(self.rnn_size, self.input_size)

	# Forward pass
	def forward(self, encoder_inputs, decoder_inputs, device):
		def loop_function(prev, i):
			return prev

		batch_size     = encoder_inputs.shape[0]
		# To pass these data through a RNN we need to switch the first two dimensions
		encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
		decoder_inputs = torch.transpose(decoder_inputs, 0, 1)
		state          = torch.zeros(batch_size, self.rnn_size).to(device)

		# Encoding
		for i in range(self.source_seq_len-1):
			# Apply the RNN cell
			state = self.cell(encoder_inputs[i], state)
			# Apply dropout in training
			state = F.dropout(state, self.dropout, training=self.training)

		outputs = []
		prev    = None
		# Decoding, sequentially
		for i, inp in enumerate(decoder_inputs):
			# Use teacher forcing?
			if prev is not None:
				inp = loop_function(prev, i)
			#inp = inp.detach()

			state  = self.cell(inp, state)
			# Output is seen as a residual to the previous value
			output = inp + self.fc1(F.dropout(state,self.dropout,training=self.training))
			outputs.append(output.view([1, batch_size, self.input_size]))
			prev = output
		outputs = torch.cat(outputs, 0)
		# Size should be batch_size x target_seq_len x input_size
		return torch.transpose(outputs, 0, 1)


	def get_batch( self, data, actions, device):
		"""Get a random batch of data from the specified bucket, prepare for step.
		Args
			data: a list of sequences of size n-by-d to fit the model to.
			actions: a list of the actions we are using
			device: the device on which to do the computation (cpu/gpu)
		Returns
			The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
			the constructed batches have the proper format to call step(...) later.
		"""

		# Select entries at random
		all_keys    = list(data.keys())
		chosen_keys = np.random.choice( len(all_keys), self.batch_size )
		# How many frames in total do we need?
		total_frames    = self.source_seq_len + self.target_seq_len
		encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
		decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
		decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

		# Generate the sequences
		for i in range( self.batch_size ):
			the_key = all_keys[ chosen_keys[i] ]
			# Get the number of frames
			n, _ = data[ the_key ].shape
			# Sample somewhere in the middle
			idx = np.random.randint(16, n-total_frames )
			# Select the data around the sampled points
			data_sel = data[ the_key ][idx:idx+total_frames ,:]
			# Add the data
			encoder_inputs[i,:,0:self.input_size] = data_sel[0:self.source_seq_len-1,:]
			decoder_inputs[i,:,0:self.input_size] = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
			decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]
		encoder_inputs  = torch.tensor(encoder_inputs).float().to(device)
		decoder_inputs  = torch.tensor(decoder_inputs).float().to(device)
		decoder_outputs = torch.tensor(decoder_outputs).float().to(device)
		return encoder_inputs, decoder_inputs, decoder_outputs


	def find_indices_srnn( self, data, action, subject ):
		"""
		Find the same action indices as in SRNN.
		See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
		"""
		# Used a fixed dummy seed, following
		# https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
		SEED = 1234567890
		rng = np.random.RandomState( SEED )

		subaction1 = 1
		subaction2 = 2

		T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
		T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
		prefix, suffix = 50, 100
		# Test is performed always on subject 5
		# Select 8 random sub-sequences (by specifying their indices)
		idx = []
		idx.append( rng.randint( 16,T1-prefix-suffix ))
		idx.append( rng.randint( 16,T2-prefix-suffix ))
		idx.append( rng.randint( 16,T1-prefix-suffix ))
		idx.append( rng.randint( 16,T2-prefix-suffix ))
		idx.append( rng.randint( 16,T1-prefix-suffix ))
		idx.append( rng.randint( 16,T2-prefix-suffix ))
		idx.append( rng.randint( 16,T1-prefix-suffix ))
		idx.append( rng.randint( 16,T2-prefix-suffix ))
		return idx

	def get_batch_srnn(self, data, action, subject, device):
		"""
		Get a random batch of data from the specified bucket, prepare for step.

		Args
		  data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
			v=nxd matrix with a sequence of poses
		  action: the action to load data from
		Returns
		  The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
		  the constructed batches have the proper format to call step(...) later.
		"""

		actions = ["directions", "discussion", "eating", "greeting", "phoning","posing", "purchases", "sitting", "sittingdown", "smoking",
		"takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

		if not action in actions:
		  raise ValueError("Unrecognized action {0}".format(action))

		frames = {}
		frames[action] = self.find_indices_srnn( data, action, subject )

		batch_size     = 8 # we always evaluate 8 sequences
		source_seq_len = self.source_seq_len
		target_seq_len = self.target_seq_len

		seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]
		encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float)
		decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float)
		decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float)

		# Compute the number of frames needed
		total_frames = source_seq_len + target_seq_len

		# Reproducing SRNN's sequence subsequence selection as done in
		# https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
		for i in range( batch_size ):
			_, subsequence, idx = seeds[i]
			idx = idx + 50

			data_sel = data[ (subject, action, subsequence, 'even') ]
			data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

			encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
			decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
			decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]
		encoder_inputs  = torch.tensor(encoder_inputs).float().to(device)
		decoder_inputs  = torch.tensor(decoder_inputs).float().to(device)
		decoder_outputs = torch.tensor(decoder_outputs).float().to(device)
		return encoder_inputs, decoder_inputs, decoder_outputs
