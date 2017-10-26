import sys
from helper import *
from collections import defaultdict
from difflib import SequenceMatcher
import pdb

class PostQuesAns:

    def __init__(self, post_title, post, post_sents, question_comment, answer):
        self.post_title = post_title
        self.post = post
        self.post_sents = post_sents
        self.question_comment = question_comment
        self.answer = answer

class PostQuesAnsGenerator:

    def __init__(self):
        self.post_ques_ans_dict = defaultdict(PostQuesAns)

    def get_diff(self, initial, final):
        s = SequenceMatcher(None, initial, final)
        diff = None
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            #print ("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
               #       (tag, i1, i2, initial[i1:i2], j1, j2, final[j1:j2]))
            if tag == 'insert':
                diff = final[j1:j2]
        if not diff:
            return None
        return diff        

    def get_diff_old(self, initial, final):
        orig_final = final
        s = SequenceMatcher(None, initial, final)
        m = s.find_longest_match(0, len(initial), 0, len(final))
        common = initial[m.a: m.a+m.size]
        while len(initial) > 1 and len(common) > 2:
            initial = initial[0:m.a] + initial[m.a+m.size:]
            final_left = final[0:m.b]
            final_right = final[m.b+m.size:]
            final = []
            if len(final_left) > 2:
                final = final_left
            if len(final_right) > 2:
                final += final_right
            s = SequenceMatcher(None, initial, final)
            m = s.find_longest_match(0, len(initial), 0, len(final))
            common = initial[m.a: m.a+m.size]
        if final == orig_final or len(final) < 3 or len(final) > 50:
            return None
        return final

    def find_right_question(self, answer, question_comment_candidates, vocab, word_embeddings):
        right_question = None
        max_similarity = 0.0
        answer_indices = get_indices(answer, vocab)
        for question_comment in question_comment_candidates:
            question_indices = get_indices(question_comment.text, vocab)
            curr_similarity = get_similarity(question_indices, answer_indices, word_embeddings)
            if curr_similarity > max_similarity:
                right_question = question_comment
                max_similarity = curr_similarity
        return right_question
        # if max_similarity > 0.4:
        #     return right_question
        # return None

    def find_first_question(self, answer, question_comment_candidates, vocab, word_embeddings):
        first_question = None
        first_date = None
        for question_comment in question_comment_candidates:
            if first_question == None:
                first_question = question_comment
                first_date = question_comment.creation_date
            else:
                if question_comment.creation_date < first_date:
                    first_question = question_comment
                    first_date = question_comment.creation_date
        return first_question
        #if not first_question:
        #    return None
        #question_indices = get_indices(first_question.text, vocab)
        #answer_indices = get_indices(answer, vocab)
        #similarity = get_similarity(question_indices, answer_indices, word_embeddings)
        #if similarity > 0.4:
        #    return first_question
        #return None

    def generate(self, posts, question_comments, posthistories, vocab, word_embeddings):
        freq_comments_after_edit = 0
        total_for_comments_after_edit = 0
        freq_resolved_after_question = 0
        total_for_resolved_after_question = 0
        for postId, posthistory in posthistories.iteritems():
            if not posthistory.edited_posts:
                continue
            if posts[postId].typeId != 1: # is not a main post
                continue
            if not posthistory.initial_post:
                continue

            comments_after_first_edit, comments_after_edit = False, False
            resolved_after_question, contains_question = False, False
            first_edit_date, first_question, first_answer = None, None, None
            for i in range(len(posthistory.edited_posts)):
                answer = self.get_diff(posthistory.initial_post, posthistory.edited_posts[i])
                if not answer:
                    continue
                else:
                    answer = remove_urls(' '.join(answer))
                    answer = answer.split()
                    if is_too_short_or_long(answer):
                        continue
                question_comment_candidates = []
                for comment in question_comments[postId]:
                    if comment.creation_date > posthistory.edit_dates[i]:
                        comments_after_edit = True
                        continue #Ignore comments added after the edit
                    else:
                        question_comment_candidates.append(comment)
                question = self.find_first_question(answer, question_comment_candidates, vocab, word_embeddings)
                # question = self.find_right_question(answer, question_comment_candidates, vocab, word_embeddings)
                if not question:
                    continue
                if first_edit_date == None or posthistory.edit_dates[i] < first_edit_date:
                    first_question, first_answer, first_edit_date = question, answer, posthistory.edit_dates[i]
                    comments_after_first_edit = comments_after_edit

            if not first_question:
                continue 
            self.post_ques_ans_dict[postId] = PostQuesAns(posts[postId].title, posthistory.initial_post, \
                                                            posthistory.initial_post_sents, first_question.text, first_answer)
            if posts[postId].accepted_answerId:
                freq_resolved_after_question += 1
            if comments_after_first_edit:
                freq_comments_after_edit += 1
            total_for_comments_after_edit += 1
            total_for_resolved_after_question += 1

        print 'Freq Comments after Edit: %.2f' % (float(freq_comments_after_edit)/total_for_comments_after_edit)
        print 'Freq Resolved after Question: %.2f' % (float(freq_resolved_after_question)/total_for_resolved_after_question)
        return self.post_ques_ans_dict
