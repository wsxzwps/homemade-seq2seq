import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np 

class CustomDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	def __init__(self, config, datafile, forceNoNoise=False): #, wordDictFile): #, labeled=True, needLabel=True):
		super(CustomDataset, self).__init__()
		print('- dataset: '+datafile)

		self.data = self.readData(datafile)

		with open(config['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp,encoding='latin1')
		self.sos_id = 2 
		self.eos_id = 3
		self.unk_id = 1
		# self.isTrans = config['isTrans']

		# self.sm = StyleMarker(config['selfatt'],self.wordDict)
		# self.sm.get_att(['the', 'service', 'was', 'really', 'good', 'too'])
		# self.sm.mark(['i', 'had', 'the', 'baja', 'burro', '...', 'it', 'was', 'heaven'])
		pass

	def readData(self,datafile):
		question = []
		response = []

		# proc .0 file (negative)
		with open(datafile, 'r') as f:
			lines = f.readlines()

		for i in range(len(lines)):
			sentences = lines[i].split('__eou__')[:-1] # there's one empty sentence in the end
		
			for j in range(len(sentences)-1):
				question.append(sentences[j].strip().split())
				response.append(sentences[j+1].strip().split())

		return question, response

	def __len__(self):
		return len(self.data[0])

	def __getitem__(self, idx):
		question_idx = self.word2index(self.data[0][idx])
		response_idx = self.word2index(self.data[1][idx])

		return (question_idx,response_idx)

	def word2index(self, sentence):
		indArr = []
		indArr.append(self.sos_id)
		for i in range(len(sentence)):
			word = sentence[i]
			if word in self.wordDict:
				indArr.append(self.wordDict[word])
			else:
				indArr.append(self.unk_id)
		indArr.append(self.eos_id) 
		indArr = np.array(indArr)
		return indArr
		
def seq_collate(batch):
	# print('>>>>>>>batch: '+str(batch))
	batchSize = len(batch)
	def extract(ind):
		maxLen = 0
		lengths = []
		for seq in batch:
			seqLen = len(seq[ind])
			lengths.append(seqLen)
			if seqLen > maxLen:
				maxLen = seqLen
		packed = np.zeros([batchSize, maxLen])
		for i in range(batchSize):
			packed[i][:lengths[i]] = batch[i][ind]
		lengths = np.array(lengths)
		# inds = np.argsort(lengths)[::-1]
		return torch.LongTensor(packed), torch.tensor(lengths)

	question, qLengths = extract(0)
	response, rLengths = extract(1) 

	return {'question': question,
			'qLengths': qLengths,
			'response': response,
			'rLengths': rLengths
		}

class LoaderHandler(object):
	"""docstring for LoaderHandler"""
	def __init__(self, config):
		super(LoaderHandler, self).__init__()
		print('loader handler...')	
		mode = config['opt'].mode
		config = config['loader']
		testData = CustomDataset(config,config['testFile'],forceNoNoise=True)
		self.ldTestEval = DataLoader(testData,batch_size=1, shuffle=False, collate_fn=seq_collate)

		trainData = CustomDataset(config,config['trainFile'])
		self.ldTrain = DataLoader(trainData,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		# elif mode == 'val':
		devData = CustomDataset(config,config['devFile'],forceNoNoise=True)
		self.ldDev = DataLoader(devData,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)
