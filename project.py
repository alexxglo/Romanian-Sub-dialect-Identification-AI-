import re
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def word_extraction(sentence):             # functie pentru sters caractere de care nu avem nevoie precum !@#$%^&*()

    words = re.sub("[^\w]", " ", sentence).split()

    cleaned_text = [w.lower()
                    for w in words]

    return cleaned_text

def get_samples(filepath):    # functie care imi ia propozitiile din fisierul train_samples.txt
    every_sample = []
    with open(filepath) as fp:      #deschid fisierul
        line = fp.readline()        # citesc linie cu linie din fisier

        while line:
            x = word_extraction(line)   #apelez functia de eliminare caractere
            y = ' '.join(x[1:])  # functie pentru imbinat cuvintele de pe line, incepand cu al doilea cuvant (primul e id-ul)
            every_sample.append(y)      # adaug propozitia curatata intr-un vector
            line = fp.readline()

    return every_sample


def get_labels(filepath2):          # pun fiecare element din train_labels intr-un vector
    for line in filepath2:
        for word in line.split():
            train_labels.append(word)
    return train_labels


def get_validationlabels(filepathx):        # pun fiecare element din validation_labels intr-un vector
    randomarray = []
    for line in filepathx:
        for word in line.split():
            randomarray.append(word)
    return randomarray

def get_validation_samples(filepath):       # pun fiecare propozitie din validation_samples intr-un vector
    val_samples = []
    with open(filepath) as fp: #deschid fisier
        line = fp.readline()    # parcurg linie cu linie

        while line:
            x = word_extraction(line)   #curat propozitia
            y = ' '.join(x[1:])  # functie pentru imbinat cuvintele de pe linie
            val_samples.append(y) # adaug propozitia in vector
            line = fp.readline()
    return val_samples


def get_test_samples(filepath):     # iau propozitiile din test_samples si le pun intr-un vector
    with open(filepath) as fp:      #deschid fisier
        line = fp.readline()        # parcurg linie cu linie

        while line:
            x = word_extraction(line)   # curat propozitia
            y = ' '.join(x[1:])  # functie pentru imbinat cuvintele de pe linie
            test_samples.append(y)  # adaug propozitia in vector
            line = fp.readline()
    return test_samples


def get_test_labels(filepath):      # functie pentru id-urile din fisier
    with open(filepath) as fp:
        line = fp.readline()

        while line:
            x = word_extraction(line)
            z = ' '.join(x[:1])     # iau doar id-ul din fisier (primul element de pe fiecare linie)
            labels.append(z)
            line = fp.readline()
    return labels


def print_results(path, labels, pred):      # functie pentru scriere in txt
    with open(path, mode='w', newline='') as sm:    # deschid fisierul
        writer = csv.writer(sm, delimiter=',')
        writer.writerow(['id', 'label'])        # scriu pe prima linie id,label
        for i in range(2623):
            writer.writerow([labels[i], pred[i]])   # scriu predictiile


train_samples = []  # vector ce contine propozitiile din train samples
test_samples = []   # vector ce contine propozitiile din test samples
train_labels = []   # vector ce contine liniile din train labels
all_labels = []     # vector ce contine toate clasificarile propozitiilor din train labels
every_sample = []   # vector ce contine toate propozitiile curatate din train samples
labels = []         # vector ce contine toate id-urile propozitiilor

filepath = "C:/Users/Aly/PycharmProjects/IA2/train_samples.txt"

filepath2 = open("C:/Users/Aly/PycharmProjects/IA2/train_labels.txt", 'r', encoding='utf8').readlines()

filepath3 = "C:/Users/Aly/PycharmProjects/IA2/test_samples.txt"

train_labels = get_labels(filepath2)

for i in range(0, len(train_labels)):
    if (i % 2 == 1):                                # iau doar elementele impare
        all_labels.append(int(train_labels[i]))     # adica doar clasificarile propozitiilor din train

every_sample = get_samples(filepath)

test_samples = get_test_samples(filepath3)

labels = get_test_labels(filepath3)


vectorizer = TfidfVectorizer()      # am ales tfidf

X = vectorizer.fit_transform(every_sample)

test = vectorizer.transform(test_samples)

# Cod pentru testarea cu elementele de test

clf = MLPClassifier(max_iter=600, activation="relu",solver="adam", hidden_layer_sizes=(250,), learning_rate="adaptive", learning_rate_init=1e-5, verbose=True, random_state=1)
# cel mai bun rezultat pe care l-am avut a fost cu MLP Classifier

clf.fit(X, all_labels)

pred = clf.predict(test)

print_results("C:/Users/Aly/PycharmProjects/IA2/sample_submission.txt", labels, pred) # apelez functia de scriere si pun calea fisierului in care doresc sa scriu

#Cod in care am testat clasificatorii cu diferiti parametrii
#X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.33, random_state=42)
#clf.fit(X_train, y_train)
#pred=clf.predict(X_test)
#print(accuracy_score(pred,y_test))


#   Cod pentru testarea cu elementele de validare

# clf = MLPClassifier(max_iter=600, activation="relu",solver="adam", hidden_layer_sizes=(250,), learning_rate="adaptive", learning_rate_init=1e-5, verbose=True, random_state=1)
#
# validation_path = "C:/Users/Aly/PycharmProjects/IA2/validation_samples.txt"
# validationlabels_path = open("C:/Users/Aly/PycharmProjects/IA2/validation_labels.txt")
# validation_samples = []
# validation_labels = []
# validation_ids = []
#
# validation_samples = get_validation_samples(validation_path)
# validation_ids = get_validationlabels(validationlabels_path)
#
# for i in range(0, len(validation_ids)):
#     if (i % 2 == 1):
#         validation_labels.append(int(validation_ids[i]))
#
#
# validationtest = vectorizer.transform(validation_samples)
# clf.fit(X, all_labels)
# pred = clf.predict(validationtest)
# print(f1_score(validation_labels, pred, average='macro'))
# print(confusion_matrix(validation_labels, pred))
