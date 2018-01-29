import sys
import numpy as np
import pdb

def evaluate_model(eval_annotations_tsv, model_predictions):
	total = 0
	correct = 0
	best_total = 0
	best = 0
	best_in9 = 0
	valid_total = 0
	valid = 0
	valid_in9 = 0
	missing_pred = 0
	for line in eval_annotations_tsv.readlines():
		splits = line.strip('\n').split('\t')
		post_id = splits[0]
		splits[1:-1] = [int(val) for val in splits[1:-1]]
		if not model_predictions.has_key(post_id):
			missing_pred += 1
			#print post_id
			continue
		is_best = splits[1:11]
		is_best_confidence = splits[11:21]
		is_valid = splits[21:]
		high_best_agreement = False
		high_valid_agreement = False
		for v in range(10):
			if is_best[v] > 1:
				high_best_agreement = True
			if is_valid[v] > 1:
				high_valid_agreement = True
		#if not high_best_agreement or not high_valid_agreement:
		#	continue	
		if np.argmax(model_predictions[post_id]) == 0:
			correct += 1
		total += 1
		if high_best_agreement:
			if is_best[np.argmax(model_predictions[post_id])] > 0:
				best += 1
				if np.argmax(model_predictions[post_id]) != 0:
					best_in9 += 1
			best_total += 1
		if high_valid_agreement:
			if is_valid[np.argmax(model_predictions[post_id])] > 0:
				valid += 1
				if np.argmax(model_predictions[post_id]) != 0:
					valid_in9 += 1
			valid_total += 1
	"""
	print 'Total: %d' % total
	print 'Accuracy: %.2f' % (correct*100.0/total)
	print 'Best Total: %d' % best_total
	print 'Accuracy in best: %.2f' % (best*100.0/total)
	print 'Accuracy in best in 9: %.2f' % (best_in9*100.0/total)
	print 'Valid Total: %d' % valid_total
	print 'Accuracy in valid: %.2f' % (valid*100.0/total)
	print 'Accuracy in valid in 9: %.2f' % (valid_in9*100.0/total)
	print 'Missing entry: %d' % missing_pred
	"""
	return correct, total, best, best_in9, best_total, valid, valid_in9, valid_total
		
def read_model_predictions(model_predictions_file):
	askubuntu_model_predictions = {}
	unix_model_predictions = {}
	for line in model_predictions_file.readlines():
		splits = line.strip('\n').split()
		sitename, post_id = splits[0][1:-2].split('_')
		if sitename not in ['askubuntu', 'unix']:
			continue
		predictions = [float(val) for val in splits[1:]]
		if sitename == 'askubuntu':
			askubuntu_model_predictions[post_id] = predictions
		else:
			unix_model_predictions[post_id] = predictions
	return askubuntu_model_predictions, unix_model_predictions

if __name__ == "__main__":
	askubuntu_eval_annotations_tsv = open(sys.argv[1], 'r')
	unix_eval_annotations_tsv = open(sys.argv[2], 'r')
	model_predictions_file = open(sys.argv[3], 'r')
	askubuntu_model_predictions, unix_model_predictions = read_model_predictions(model_predictions_file)
	print 'askubuntu.com'
	a_correct, a_total, a_best, a_best_in9, a_best_total, a_valid, a_valid_in9, a_valid_total = evaluate_model(askubuntu_eval_annotations_tsv, askubuntu_model_predictions)	
	print 'unix'
	u_correct, u_total, u_best, u_best_in9, u_best_total, u_valid, u_valid_in9, u_valid_total = evaluate_model(unix_eval_annotations_tsv, unix_model_predictions)
	print 'aggregate'
	correct = a_correct + u_correct
	total = a_total + u_total
	best = a_best + u_best
	best_in9 = a_best_in9 + u_best_in9
	best_total = a_best_total + u_best_total
	valid = a_valid + u_valid
	valid_in9 = a_valid_in9 + u_valid_in9
	valid_total = a_valid_total + u_valid_total
	print 'Correct %d' % correct
	print 'Best %d' % best
	print 'Best in 9 %d' % best_in9
	print 'Valid %d' % valid
	print 'Valid in 9 %d' % valid_in9
	print 'Total %d' % total
	print 
	print 'Accuracy: %.2f' % (correct*100.0/total)
	print 'Best Total: %d' % best_total
	print 'Accuracy in best: %.2f' % (best*100.0/best_total)
	print 'Accuracy in best in 9: %.2f' % (best_in9*100.0/best_total)
	print 'Valid Total: %d' % valid_total
	print 'Accuracy in valid: %.2f' % (valid*100.0/valid_total)
	print 'Accuracy in valid in 9: %.2f' % (valid_in9*100.0/valid_total)
	
