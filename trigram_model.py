import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
Trigram Language Models

HW 1 COMS W4705
Brian Yang - by2289

"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    # sequence - list of words
    # n - number of grams

    ngrams = list()

    # insert START and STOP to beginning and end of the sequence
    # copy the sequence into a new list so we do not modify the original
    # since the original is passed by reference
    copy = sequence.copy()
    copy.insert(0, "START")
    copy.append("STOP")

    if n < 1:
        return None
    elif n == 1:
        # ("START",) unigram must be included
        right = 0
    else:
        right = 1

    while right < len(copy):
        left = right - n + 1
        if left < 0:
            # must pad with "START"
            ngram = list()

            # insert abs(left) number of "START"s
            for i in range(abs(left)):
                ngram.append("START")

            # insert remaining elements to form n-gram
            for i in range(right + 1):
                ngram.append(copy[i])

        else:
            ngram = list()
            for i in range(left, right + 1):
                ngram.append(copy[i])

        # convert ngram into tuple and add to list of ngrams
        ngram = tuple(ngram)
        ngrams.append(ngram)

        right += 1

    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # Iterate through the corpus to count total number of words
        # in the document after normalization
        generator = corpus_reader(corpusfile, self.lexicon)
        self.total_num_words = 0
        for sentence in generator:
            self.total_num_words += len(sentence) + 1  # +1 for "STOP" for every sentence

        self.num_distinct_words = len(self.unigramcounts.keys()) - 1  # - 1 for ("START",) unigram

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = {}  # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        ##Your code here
        num_sentences = 0
        for sequence in corpus:
            num_sentences += 1

            # use a loop to fill in uni,bi,and tri grams
            n = 1
            while n <= 3:

                ngrams = get_ngrams(sequence, n)

                # add update counts of the n-grams in the dictionary
                for ngram in ngrams:
                    if n == 1:
                        if ngram not in self.unigramcounts:
                            self.unigramcounts[ngram] = 1
                        else:
                            self.unigramcounts[ngram] += 1
                    elif n == 2:
                        if ngram not in self.bigramcounts:
                            self.bigramcounts[ngram] = 1
                        else:
                            self.bigramcounts[ngram] += 1
                    elif n == 3:
                        if ngram not in self.trigramcounts:
                            self.trigramcounts[ngram] = 1
                        else:
                            self.trigramcounts[ngram] += 1
                n += 1

        # include ("START",) and ("START", "START") for bigram and
        # trigram probability calculations later on
        self.unigramcounts[("START",)] = num_sentences
        self.bigramcounts[("START", "START")] = num_sentences

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        # given a trigram (u,v,w), the probability of the trigram
        # is given by
        #
        # P(w | u, v) = count(u, v, w) / count(u, v)
        #
        # count(u, v, w) = trigram count for (u,v,w)
        # count(u, v) = bigram count for (u,v)

        bigram = (trigram[0], trigram[1])
        if trigram in self.trigramcounts and bigram in self.bigramcounts:

            count_u_v_w = self.trigramcounts[trigram]
            count_u_v = self.bigramcounts[bigram]
            prob = count_u_v_w / count_u_v

        else:
            prob = 0.0

        return prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # given a bigram (v,w), the probability of the bigram
        # is given by
        #
        # P(w | v) = count(v, w) / count(v)
        #
        # count(v, w) = bigram count for (v,w)
        # count(v) = unigram count for (v)

        unigram = (bigram[0],)
        if bigram in self.bigramcounts and unigram in self.unigramcounts:

            count_v_w = self.bigramcounts[bigram]
            count_v = self.unigramcounts[unigram]
            prob = count_v_w / count_v

        else:
            prob = 0.0

        return prob

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.

        # P(w) = count(w)/N
        # N = number of tokens

        if unigram in self.unigramcounts:

            count_w = self.unigramcounts[unigram]
            N = self.total_num_words
            prob = count_w / N

        else:
            prob = 0.0

        return prob

    '''
    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result
    '''

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0

        # trigram input format: (u, v, w)

        # Linear Interpolation
        # p(w | u, v) = lambda1 * p_MLE(w | u, v) + lambda2 * p_MLE(w | v) + lambda3 * p_MLE(w)

        # p_MLE(w | u, v) - (u, v, w)
        p_MLE_w_u_v = self.raw_trigram_probability(trigram)

        # p_MLE(w | v) - (v, w)
        bigram = (trigram[1], trigram[2])
        p_MLE_w_v = self.raw_bigram_probability(bigram)

        # p_MLE(w) - (w,)
        unigram = (trigram[2],)
        p_MLE_w = self.raw_unigram_probability(unigram)

        # p(w | u, v)
        prob = lambda1 * p_MLE_w_u_v + lambda2 * p_MLE_w_v + lambda3 * p_MLE_w

        return prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        log_prob = 0.0

        # Step 1:
        #   Get trigrams for sentence
        trigrams = get_ngrams(sentence, 3)

        # Step 2:
        #   For each trigram (w_i-2, w_i-1, w_i), find P(w_i | w_i-2, w_i-1)
        #   for i = 1,2,...,len(sentence)

        for trigram in trigrams:
            trigram_prob = self.smoothed_trigram_probability(trigram)
            log_prob += math.log2(trigram_prob)

        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """

        # Perplexity = 2^(-L)
        # L = (1/M) * sum_(i=1)^(m) [log_2(p(s_i))]
        # M = number of tokens in the corpus
        # m = number of sentences in the corpus

        M = 0
        sum_log_prob = 0.0

        for sentence in corpus:
            # compute sum of log probabilities
            log_prob = self.sentence_logprob(sentence)
            sum_log_prob += log_prob

            # compute total number of words
            M = M + len(sentence) + 1  # +1 for "STOP" for every sentence

        L = sum_log_prob / M
        perplexity = math.pow(2, -L)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0.0
    correct = 0.0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

        # since model1 was trained on training_file1, pp1 should be lower than
        # pp2 to be correct prediction
        if pp1 <= pp2:
            correct += 1
        total += 1

    for f in os.listdir(testdir2):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

        # since model2 was trained on training_file2, pp2 should be lower than
        # pp1 to be correct prediction
        if pp2 <= pp1:
            correct += 1
        total += 1

    return correct / total


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])
    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    print("############################################################")
    print("#        PART 1: Extracting n-grams from a Sentence        #")
    print("############################################################")

    sequence = ["natural", "language", "processing", "?"]
    print("Sequence: ", sequence)
    for n in range(1, 4):
        if n == 1:
            print("unigrams:")
        elif n == 2:
            print("bigrams:")
        elif n == 3:
            print("trigrams:")
        ngrams = get_ngrams(sequence, n)
        for ngram in ngrams:
            print(ngram)
        print()

    print("\n\n\n")

    print("######################################################")
    print("#        PART 2: Counting n-grams in a Corpus        #")
    print("######################################################")

    print("model.trigramcounts[('START','START','the')]\n", model.trigramcounts[('START', 'START', 'the')])
    print("model.bigramcounts[('START','the')]\n", model.bigramcounts[('START', 'the')])
    print("model.unigramcounts[('the',)]\n", model.unigramcounts[('the',)])

    print("\n\n\n")

    print("##################################################")
    print("#        PART 3: Raw n-gram Probabilities        #")
    print("##################################################")

    trigram = ('START', 'START', 'the')
    bigram = ('START', 'the')
    unigram = ('the',)
    print(trigram, ": ", model.raw_trigram_probability(trigram))
    print(bigram, ": ", model.raw_bigram_probability(bigram))
    print(unigram, ": ", model.raw_unigram_probability(unigram))
    print("\n\n\n")

    print("###############################################")
    print("#       PART 4: Smoothed Probabilities        #")
    print("###############################################")

    trigram = ('START', 'START', 'the')
    print(trigram, ": ", model.smoothed_trigram_probability(trigram))
    print("\n\n\n")

    print("#######################################################")
    print("#       PART 5: Computing Sentence Probability        #")
    print("#######################################################")

    sentence = ["natural", "UNK", "language", "processing", "?"]
    print("Sentence: ", sentence)
    print("Log probability: ", model.sentence_logprob(sentence))
    print("\n\n\n")

    print("###################################")
    print("#       PART 6: Perplexity        #")
    print("###################################")

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)

    pp = model.perplexity(dev_corpus)
    pp_train = model.perplexity(corpus_reader(sys.argv[1], model.lexicon))
    print("Perplexity of testing data = ", pp)
    print("Perplexity of training data = ", pp_train)

    print("\n\n\n")

    print("################################################################")
    print("#       PART 7: Using the Model for Text Classification        #")
    print("################################################################")

    # Essay scoring experiment:

    train_high_txt = "hw1_data/ets_toefl_data/train_high.txt"
    train_low_txt = "hw1_data/ets_toefl_data/train_low.txt"
    test_high = "hw1_data/ets_toefl_data/test_high"
    test_low = "hw1_data/ets_toefl_data/test_low"

    acc = essay_scoring_experiment(train_high_txt, train_low_txt, test_high, test_low)
    print("Accuracy: ", acc)

    print("\n\n\n")
