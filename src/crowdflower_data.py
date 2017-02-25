import sys, os
# from BeautifulSoup import BeautifulSoup
# import xml.etree.ElementTree as ET

if __name__ == '__main__':
    dataset_file = open(sys.argv[1], 'r')
    output_file = open(sys.argv[2], 'w')
    # post_xml_file = sys.argv[3]
    
    data = {}
    posts = {}
    for line in dataset_file.readlines():
        if 'Id:' in line:
            index = line.split(':')[1].strip('\n').strip()           
        if 'Question:' in line and index:
            data[index] = line.split(':', 1)[1].strip('\n')
            index = None
        
    # posts_tree = ET.parse(post_xml_file)
    # for post in posts_tree.getroot():
    #     postId = post.attrib['Id']
    #     try:
    #         data[postId]
    #         post = post.attrib['Body']
    #         post = BeautifulSoup(post.encode('utf-8').decode('ascii', 'ignore')).text
    #         posts[postId] = post
    #     except:
    #         continue
    output_file.write('%s\t%s\n' % ("index", "sentence"))
    # output_file.write('%s\t%s\t%s\n' % ("index", "sentence", "post"))
    for index in data.keys():
        sentence = data[index]
        output_file.write('%s\t%s\n' % (index, sentence))
        # post = posts[index]
        # output_file.write('%s\t%s\t%s\n' % (index, sentence, post))
    
    
        
    
