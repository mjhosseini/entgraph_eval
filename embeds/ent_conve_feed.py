#DEPRECATED
import sys
sys.path.append("..")
import embed_reader as embed
import e_util

from evaluation import util

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Hyper-parameters
input_size = 601
hidden_size = 300
num_classes = 2
num_epochs = 20
learning_rate = 0.001

print ("input size: ", input_size)

def getExampleVectorsAndLabels(ccgpath, fname_orig, er):
    cos_sims = []
    data = util.read_data(ccgpath,fname_orig,True,False,False)

    ret = []
    not_covered = 0

    for p, q, _, _, _, a, l in data:
        if p!="" and q!="":
            if embMode=="glove":
                phrase1 = e_util.getPhraseFromCCGRel(p)
                phrase2 = e_util.getPhraseFromCCGRel(q)
            else:
                phrase1 = p
                phrase2 = q

                print ("a is: ", a)

                if embMode=="convE" and not a:
                    phrase2 += "_reverse"


            emb1 = er.getEmbeddingOfPhrase(phrase1)
            emb2 = er.getEmbeddingOfPhrase(phrase2)

            dim = len(emb1)
            emb1_nd = np.zeros(shape=(1,dim))
            emb2_nd = np.zeros(shape=(1, dim))
            emb3_nd = np.zeros(shape=(1, dim))
            emb4_nd = np.zeros(shape=(1, 1))

            if sum(emb1)==0 or sum(emb2)==0:
                not_covered += 1
                # this_vect = np.zeros(shape=(1, 3 * dim+1))  # I know the first time dim will get initialized :)
            # else:
            emb1_nd[0,:] = emb1
            emb2_nd[0,:] = emb2
            emb3_nd[0,:] = emb1_nd[0,:] - emb2_nd[0,:]
            emb4_nd[0,:] = np.array([(cosine_similarity(emb1_nd, emb2_nd)[0, 0] + 1)/2])

            cos_sims.append((cosine_similarity(emb1_nd, emb2_nd)[0, 0] + 1)/2)

            this_vect = np.concatenate((emb1_nd,emb2_nd,emb3_nd,emb4_nd),axis=1)
            # this_vect = np.concatenate((emb1_nd, emb2_nd), axis=1)
            # this_vect = np.concatenate((emb3_nd, emb4_nd), axis=1)
            # print (this_vect.shape)

        else:
            this_vect = np.zeros(shape=(1,input_size))#I know the first time dim will get initialized :)
            not_covered += 1
            cos_sims.append(0)

        ret.append((this_vect,l))
    print ("all: ", len(ret), " not covered: ", not_covered)

    return ret, cos_sims


root = "../../gfiles/"

# python3 ent_conve_feed.py convE rels2emb_ConvE_NS_unt_20_20_125.txt rels_NS_10_10 none dev_new_rels.txt dev_new.txt trainTest_new_rels.txt trainTest_new.txt trainTest_convE_feed.txt
#python3 ent_conve_feed.py convE rels2emb_ConvE_NS_unt_10_10_1000.txt rels_NS_10_10 none conf_rels_100k.txt conf_rels_100k.txt dev_new_rels.txt dev_new.txt dev_convE_feed_conf.txt
#python3 ent_conve_feed.py convE rels2emb_ConvE_NS_unt_10_10_1000.txt rels_NS_10_10 none conf_rels_100k.txt conf_rels_100k.txt dev_new_rels.txt dev_new.txt dev_convE_feed_conf.txt

if len(sys.argv) < 9:
    print ("usage: ent_conve_feed.py <embMode> <emb-datafile> <all-rels-file> <relsFolder> <ent-examples-train-file> <ent-examples-train-orig-file> <ent-examples-test-file> <ent-examples-test-orig-file> <ent-examples-file-out>")
    exit()


embMode = sys.argv[1]
embedPath = sys.argv[2]
relspath = sys.argv[3]
relsFolder = sys.argv[4]
train_fname_CCG = root+"ent/"+sys.argv[5]
train_fname_orig = root+"ent/"+sys.argv[6]
test_fname_CCG = root+"ent/"+sys.argv[7]
test_fname_orig = root+"ent/"+sys.argv[8]
out_path = sys.argv[9]

if relspath!="":
    try:
        allRelsTotal = e_util.loadAllrelsTotal(relspath)
    except:
        allRelsTotal = None

else:
    allRelsTotal = None

er = embed.EmbedReader(embedPath, 400000, embMode ,allRelsTotal, relsFolder)
# fname_CCG = root+"ent/trainTest_new_rels.txt"
# ber_all_rels.txt"

train_vectors_labels, _ = getExampleVectorsAndLabels(train_fname_CCG, train_fname_orig, er)
shuffle(train_vectors_labels)
test_vectors_labels, cos_sims = getExampleVectorsAndLabels(test_fname_CCG, test_fname_orig, er)
print ("len cos sims:", len(cos_sims))




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
        # return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# weights = [1,2]
# class_weights = torch.FloatTensor(weights).cuda()
# Loss and optimizer
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = .01)

# Train the model
total_step = len(train_vectors_labels)

for epoch in range(num_epochs):
    total_loss = 0
    for i, (vect,l) in enumerate(train_vectors_labels):
        # Move tensors to the configured device
        vect_tensor = torch.tensor(vect,dtype=torch.float).to(device)
        # print (type(vect_tensor))
        # print (vect_tensor)
        label = torch.tensor([l],dtype=torch.long).to(device)#float(l).to(device)
        # if l==0 and random.random() < .6:
        #     continue

        # Forward pass
        output = model(vect_tensor)
        # print (output)
        # print (type(output))
        # print (label)
        # print (type(label))
        loss = criterion(output, label)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # if (i + 1) % 100 == 0:
        #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    print ("total loss:", total_loss)


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

entScores = []

with torch.no_grad():
    correct = 0
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, (vect, l) in enumerate(test_vectors_labels):
        vect_tensor = torch.tensor(vect, dtype=torch.float).to(device)

        output = model(vect_tensor)
        entScore = output.data.cpu().numpy()[0][1]
        entScore = np.exp(entScore)
        print (entScore)
        entScores.append(entScore)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.data.cpu().numpy()[0]
        total += 1
        print (predicted)
        print (l)
        correct += (predicted == l)
        print ("predicted: ", predicted)

        if predicted==1:
            if predicted==l:
                tp += 1
            else:
                fp += 1
        else:

            if predicted == l:
                tn += 1
            else:
                fn += 1


        # print ("correct is: ", correct)

    print (correct)
    print (total)

    print('Accuracy of the network on the test entailments: {} %'.format(100 * correct / total))
    print ("precision: ", (tp/(tp+fp)))
    print ("recall: ", tp/(tp + fn))

    print ("conf matrix:")
    print ("tp: ", tp)
    print ("fp: ", fp)
    print ("tn: ", tn)
    print ("fn: ", fn)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

# print (entScores)

f = open(out_path,'w')

for s in entScores:
# for s in cos_sims:
    print (s)
    f.write(str(s)+"\n")
f.close()