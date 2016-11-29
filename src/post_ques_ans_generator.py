import sys
from helper import *

class PostQuesAns:

	def __init__(self, post, question_comment, answer):
		self.postId = post
		self.question_comment = question_comment
		self.answer = answer

class PostQuesAnsGenerator:

	def __init__(self):
		self.post_ques_answer_dict = defaultdict(PostQuesAns)

	def get_diff(self, initial, final):
		orig_final = final
		s = SequenceMatcher(None, initial, final)
		m = s.find_longest_match(0, len(initial), 0, len(final))
		common = initial[m.a: m.a+m.size]
		while ' ' in initial and len(common.split()) > 2:
			initial = initial[0:m.a] + ' ' + initial[m.a+m.size:]
			final_left = final[0:m.b]
			final_right = final[m.b+m.size:]
			final = ''
			if len(final_left.split()) > 2:
				final = final_left + ' '
			if len(final_right.split()) > 2:
				final += final_right
			s = SequenceMatcher(None, initial, final)
			m = s.find_longest_match(0, len(initial), 0, len(final))
			common = initial[m.a: m.a+m.size]
		if final == orig_final or final.split() < 3 or final.split() > 50:
			return None
		return final

	def find_right_question(self, answer, question_candidates, vocab, word_embeddings):
		right_question = None
		max_similarity = 0.0
		answer_toks = helper.get_tokens(answer)
		answer_indices = get_indices(answer_toks, vocab)
		for question in question_candidates:
			question_toks = helper.get_tokens(question)
			question_indices = helper.get_indices(question_toks, vocab)
			curr_similarity = helper.get_similarity(question_indices, answer_indices, word_embeddings)
			if curr_similarity > max_similarity:
				right_question = question
				max_similarity = curr_similarity
		if max_similarity > 0.4:
			return right_question
		return None

	def generate(self, posts, posthistories, question_comments, word_embeddings):
		for postId, posthistory in posthistories.iteritems():
			if not posthistory.edited_post:
				continue
			if posts[postId].typeId != '1': # is not a main post
				continue
			answer = self.get_diff(posthistory.initial_post, posthistory.edited_post)
			if not answer:
				continue
			question_candidates = question_comments[postId]
			if not question_candidates:
				continue
			question = self.find_right_question(answer, question_candidates, word_embeddings)
			if not question:
				continue
			self.post_ques_ans_dict[postId] = PostQuesAns(posthistory.initial_post, helper.get_tokens(question), helper.get_tokens(answer))
		return self.post_ques_ans_dict


