import sys
import argparse
from parse import *
from post_ques_ans_generator import * 
from helper import *

def main(args):
	post_parser = PostParser(args.posts_xml)
	post_parser.parse()
	posts = post_parser.get_posts()

	comment_parser = CommentParser(args.comments_xml)
	comment_parser.parse()
	question_comments = comment_parser.get_question_comments()

	posthistory_parser = PostHistoryParser(args.posthistory_xml)
	posthistory_parser.parse()
	posthistories = posthistory_parser.get_posthistories()

	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)

	post_ques_ans_generator = PostQuesAnsGenerator()
	post_ques_answers = post_ques_ans_generator.generate(posts, question_comments, posthistories, word_embeddings)

	post_vectors = [None]*len(post_ques_answers)
	ques_list_vectors = [None]*len(post_ques_answers)
	ans_list_vectors = [None]*len(post_ques_answers)
	vocab = p.load(open(args.vocab, 'rb'))
	similar_posts = p.load(open(args.similar_posts), 'rb')

	i = 0
	N = args.no_of_candidates
	for postId in post_ques_answers.keys():
		post_vectors[i] = get_indices(post_ques_answers[postId].post)
		ques_list_vectors[i] = [None]*N
		ques_list_vectors[i][0] = get_indices(post_ques_answers[postId].ques)
		ans_list_vectors[i] = [None]*N
		ans_list_vectors[i][0] = get_indices(post_ques_answers[postId].ans)
		candidate_postIds = similar_posts[postId][:N-1]
		for j in range(N-1):
			ques_list_vectors[i][j+1] = post_ques_answers[candidate_postIds[j]].ques
			ans_list_vectors[i][j+1] = post_ques_answers[candidate_postIds[j]].ans
		i+=1

	p.dump(post_vectors, open(args.post_vectors, 'wb'))
	p.dump(ques_list_vectors, open(args.ques_list_vectors, 'wb'))
	p.dump(ans_list_vectors, open(args.ans_list_vectors, 'wb'))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--posts_xml", type = str)
	argparser.add_argument("--comments_xml", type = str)
	argparser.add_argument("--posthistory_xml", type = str)
	argparser.add_argument("--post_vectors", type = str)
	argparser.add_argument("--ques_list_vectors", type = str)
	argparser.add_argument("--ans_list_vectors", type = str)
	argparser.add_argument("--similar_posts", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--no_of_candidates", type = int, default = 5)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

