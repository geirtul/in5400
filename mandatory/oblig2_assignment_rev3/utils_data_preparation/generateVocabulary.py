import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import trange

# -------------------------------------------- Generate vocabulary -----------------------------------------------------
def generateVocabulary(data_dir, train_records_list, test_records_list):
    # Init
    vocabulary  = []
    wordCounter = []
    sentenceLen = []
    listOfSentences = []

    # Go through all test data
    for ii in trange(len(test_records_list), desc='Generate: Test records', leave=True):
        # print(ii)
        #t.set_description("Bar desc (file %i)" % ii)
        for kk in range(len(test_records_list[ii][2])):
            sentence = test_records_list[ii][2][kk]
            listOfSentences.append(sentence)

            #Convert all characters to lowercase
            sentence = sentence.lower()

            #Spilt sentence to list of words
            sentence_list = sentence.split(' ')

            # Add sentence length to a list
            sentenceLen.append(len(sentence_list))

            #Remove all special characters and add to vocabulary
            for jj in range(len(sentence_list)):
                word = sentence_list[jj]
                word = ''.join(e for e in word if e.isalnum())

                if vocabulary.count(word)==0:
                    vocabulary.append(word)
                    wordCounter.append(1)
                else:
                    id = vocabulary.index(word)
                    wordCounter[id] = wordCounter[id] + 1

    # Go through all training data
    for ii in trange(len(train_records_list), desc='Generate: Train records', leave=True):
        for kk in range(len(train_records_list[ii][2])):
            sentence = train_records_list[ii][2][kk]
            listOfSentences.append(sentence)

            #Convert all characters to lowercase
            sentence = sentence.lower()

            #Spilt sentence to list of words
            sentence_list = sentence.split(' ')

            # Add sentence length to a list
            sentenceLen.append(len(sentence_list))

            #Remove all special characters and add to vocabulary
            for jj in range(len(sentence_list)):
                word = sentence_list[jj]
                word = ''.join(e for e in word if e.isalnum())

                if word != '':
                    if vocabulary.count(word)==0:
                        vocabulary.append(word)
                        wordCounter.append(1)
                    else:
                        id = vocabulary.index(word)
                        wordCounter[id] = wordCounter[id] + 1


    # --------------------------------------------- Process statistics -----------------------------------------------------
    wordCounter = np.asanyarray(wordCounter)
    sentenceLen = np.asarray(sentenceLen)

    figShape = (20,10)
    labelSize = 24
    tickSize  = 16
    plotFig = False
    # ------------ Sequence lengths (histogram) ----------
    unique, counts = np.unique(sentenceLen, return_counts=True)
    if plotFig:
        plt.figure(figsize=figShape)
        plt.bar(left=unique, height=counts, width=1.0)
        plt.xlabel('Caption length', fontsize=labelSize)
        plt.ylabel('Counts [#]', fontsize=labelSize)
        plt.tick_params(axis='both', which='major', labelsize=tickSize)
        plt.xlim([0, 40])
        plt.savefig('utils_images/Sequence_lengths_hist.png')

    # ------------ Captured sentences as a function of sequence length --------
    x_vals = np.linspace(0,unique.max(),unique.max()+1)
    bincounts = np.bincount(sentenceLen)
    countsCum = np.cumsum(bincounts)
    numbOfSentences = np.sum(counts)

    countsCumRel = countsCum/numbOfSentences*100

    if plotFig:
        plt.figure(figsize=figShape)
        plt.bar(left=x_vals, height=countsCumRel, width=1.0)
        plt.ylim([60, 102])
        plt.xlim([0, 25])
        plt.xlabel('Caption length', fontsize=labelSize)
        plt.ylabel('Percentage of all captions [%]', fontsize=labelSize)
        plt.tick_params(axis='both', which='major', labelsize=tickSize)
        plt.savefig('utils_images/accumulated_Sequence_length_count.png')

    # ----------- vocabulary statistics ----------------------------------------
    numbOfWords = len(wordCounter)
    ids = np.flipud(np.argsort(wordCounter))

    wordCounterSort = wordCounter[ids]
    vocabularySort = [vocabulary[i] for i in ids]

    subFactor=100
    numbOfWordsSub = int(np.ceil(numbOfWords/subFactor))
    x_vals             = [(i*subFactor)+1 for i in range(numbOfWordsSub)]
    wordCounterSortSub = [sum(wordCounterSort[i*subFactor:(i+1)*subFactor]) for i in range(numbOfWordsSub)]
    #wordCounterSortSub.append(  wordCounterSort[(numbOfWordsSub-1)*10:-1]  )

    x_vals = np.asarray(x_vals)
    wordCounterSortSub = np.asarray(wordCounterSortSub)

    if plotFig:
        plt.figure(figsize=figShape)
        plt.bar(left=x_vals, height=wordCounterSortSub/1e6, width=100)
        plt.xlabel('Word index', fontsize=labelSize)
        plt.ylabel('Word count in millions (bin size=%d)' % subFactor, fontsize=labelSize)
        plt.tick_params(axis='both', which='major', labelsize=tickSize)
        plt.savefig('utils_images/Word_count_hist.png')


    #---------------------------------- accululative --------------------------
    countsCum = np.cumsum(wordCounterSortSub)
    numbOfSentences = np.sum(wordCounterSortSub)

    countsCumRel = countsCum/numbOfSentences*100

    if plotFig:
        id5  = np.where(x_vals==4901)[0][0]
        id10 = np.where(x_vals==9901)[0][0]
        id15 = np.where(x_vals==14901)[0][0]

        plt.figure(figsize=figShape)
        plt.bar(left=x_vals, height=countsCumRel, width=100)
        plt.ylim([98, 100])
        plt.xlim([0, numbOfWords])
        plt.xlabel('Vocabulary size', fontsize=labelSize)
        plt.ylabel('Percentage of all words [%]', fontsize=labelSize)
        plt.plot([x_vals[id5+1], x_vals[id5+1]], [0, countsCumRel[id5]], color='r', linestyle='--')
        plt.plot([0, x_vals[id5+1]], [countsCumRel[id5], countsCumRel[id5]], color='r', linestyle='--')
        plt.plot([x_vals[id10+1], x_vals[id10+1]], [0, countsCumRel[id10]], color='g', linestyle='--')
        plt.plot([0, x_vals[id10+1]], [countsCumRel[id10], countsCumRel[id10]], color='g', linestyle='--')
        plt.plot([x_vals[id15+1], x_vals[id15+1]], [0, countsCumRel[id15]], color='m', linestyle='--')
        plt.plot([0, x_vals[id15+1]], [countsCumRel[id15], countsCumRel[id15]], color='m', linestyle='--')
        plt.tick_params(axis='both', which='major', labelsize=tickSize)
        plt.savefig('utils_images/accumulated_Word_count.png')


    # ---------------------------------------- Store vocabulary dict  -----------------------------------------------

    #We add these to the vocabulary
    #'UNK' - undefined word
    #'ssss' - start token
    #'eeee' - end token

    vocabularySort.insert(0, 'UNK')
    vocabularySort.insert(0, 'ssss')
    vocabularySort.insert(0, 'eeee')
    numbOfWords = numbOfWords + 3
    wordCounter = np.insert(wordCounter, 0, values=[0,0,0])
    tokens = np.linspace(0, numbOfWords-1, numbOfWords, dtype=int)

    vocabularyDict = {}
    vocabularyDict['wordCounter'] = wordCounter
    vocabularyDict['sentenceLen'] = sentenceLen
    vocabularyDict['vocabulary']  = vocabularySort
    vocabularyDict['tokens']      = tokens

    wordToToken = dict(zip(vocabularySort, list(tokens)))
    TokenToWord = dict(zip(list(tokens), vocabularySort))
    vocabularyDict['wordToToken'] = wordToToken
    vocabularyDict['TokenToWord'] = TokenToWord

    data_dir = data_dir + 'vocabulary'
    filename = data_dir + '/vocabulary.pickle'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print('directory created')
        with open(filename, 'wb') as handle:
            pickle.dump(vocabularyDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('"vocabulary.pickle" saved')
    else:
        if os.path.isfile(filename):
            os.remove(filename)
            print('Old "vocabulary.pickle" file deleted')
        with open(filename, 'wb') as handle:
            pickle.dump(vocabularyDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('"vocabulary.pickle" saved')

def loadVocabulary(data_dir):
    filename = data_dir + 'vocabulary/vocabulary.pickle'
    with open(filename, "rb") as input_file:
        vocabularyDict = pickle.load(input_file)
    return vocabularyDict

if __name__ == "__main__":
    a = 1
    # Create dataClass
    # myImgCocoDataClass = coco.CocoImagesDataClass()
    #
    # # Load train ans test records
    # myImgCocoDataClass.load_records(trainSet=True)
    # myImgCocoDataClass.load_records(trainSet=False)
    #
    # data_dir           = myImgCocoDataClass.data_dir
    # train_records_list = myImgCocoDataClass.train_records_list
    # test_records_list  = myImgCocoDataClass.val_records_list
    #
    # generateVocabulary(data_dir, train_records_list, test_records_list)
