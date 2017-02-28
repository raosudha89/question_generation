import sys
import random

def shuffle_data(q, l):
	sq = [None]*len(q)
	sl = [None]*len(l)
	indexes = range(len(q))
	random.shuffle(indexes)
	for i, index in enumerate(indexes):
		sq[i] = q[index]
		sl[i] = l[index]
		
	return sq, sl 

if __name__ == '__main__':
	crowdflower_labels_file = open(sys.argv[1], 'r')
	vw_file = open(sys.argv[2], 'w')
	i = 0
	questions = []
	labels = []
	for line in crowdflower_labels_file.readlines():
		if i == 0:
			i += 1
			continue
		splits = line.split(',')
		#questions.append(splits[-1].strip('\n')[2:-2])
		questions.append(splits[-1].strip('\n').replace(':',''))
		if splits[1] == 'clarification_question':
			labels.append(1)
		else:
			labels.append(0)
	
	questions, labels = shuffle_data(questions, labels)
		
	for i in range(len(questions)):
		vw_line = '%d | %s' % (labels[i], questions[i])
		vw_file.write(vw_line + '\n')
		
