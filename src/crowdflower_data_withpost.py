import sys, os
import xml.etree.ElementTree as ET
import csv

if __name__ == '__main__':
    dataset_file = open(sys.argv[1], 'r')
    post_xml_file = sys.argv[2]
    output_file = sys.argv[3]
    
    data = {}
    for line in dataset_file.readlines():
        if 'Id:' in line:
            index = line.split(':')[1].strip('\n').strip()
        if 'Question:' in line and index:
            data[index] = [None, None]
            data[index][0] = line.split(':', 1)[1].strip('\n')
            index = None   
    
    posts_tree = ET.parse(post_xml_file)
    for post in posts_tree.getroot():
        postId = post.attrib['Id']
        try:
            data[postId][1] = post.attrib['Body'].encode('utf-8')
        except:
            continue
        
    # with open(output_file, 'w') as csvfile:
    #     fieldnames = ['index', 'sentence', 'post']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     #writer.writeheader()
    #     writer.writerow({'index': 'index', 'sentence': 'sentence', 'post': 'post'})
    #     for index in data.keys():
    #         sentence, post = data[index]
    #         writer.writerow({'index': index, 'sentence': sentence, 'post': post})
    
    output_file.write('%s\t%s\n' % ("index", "sentence", "post"))
    for index in data.keys():
        sentence, post = data[index]
        output_file.write("{}\t{}\t{}".format(index, sentence, post))
    
    
        
    
