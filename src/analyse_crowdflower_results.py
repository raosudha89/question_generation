import sys, pdb

if __name__ == '__main__':
    results = open(sys.argv[1], 'r')
    categories = ['clarification_question', \
                  'providing_an_answer_or_a_suggestion_even_if_phrased_as_a_rhetorical_question', \
                  'neither']
    counts = [0, 0, 0]
    confidences = [0, 0, 0]
    num = 0
    for line in results.readlines():
        if num == 0:
            num += 1
            continue
        splits = line.strip('\n').split(',', 5)
        category, confidence, index = splits[1:4]
        counts[categories.index(category)] += 1
        confidences[categories.index(category)] += float(confidence)
	num += 1    

    for i in range(3):
        print 'Category: ', categories[i]
        print 'Count: %d (%d' % (counts[i], 100*round(counts[i]*1.0/num, 2)) + '%)'
        if counts[i]:
            print 'Confidence: %.2f' % round(confidences[i]*1.0/counts[i], 2)
        else:
            print 'Confidence:', counts[i]
                
        
