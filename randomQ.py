"""
In this file, we apply the Grounhog attack to the Census 1% Teaching file of
the England and Wales census and the Raw generator.

The goal of TAPAS is to evaluate the possibility of an attack, rather than
train and deploy attacks against real datasets. This informs the design
decisions made, especially as relates to the auxiliary knowledge.

This example is meant as a general introduction to TAPAS, and explains some
important design choices to make when using the toolbox.

"""
import json

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report
from tapas.datasets.dataset import _parse_csv
# from reprosyn.methods import Dataset
from reprosyn.methods import MST
from reprosyn.methods import PRIVBAYES
# Some fancy displays when training/testing.
import tqdm

from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
# We attack the 1% Census Microdata file, available at:
#  https://www.ons.gov.uk/census/2011census/2011censusdata/censusmicrodata/microdatateachingfile
# We have created a .json description file, so that tapas.Dataset.read can load both.
data = tapas.datasets.TabularDataset.read(
    "./tapas_test/snake", label="SNAKE"
           # "/Users/golobs/Documents/GradSchool/SNAKE/", label="SNAKE"
    # "data/files/", label="SNAKE"
)
# Select an arbitrary target record (that is unique), and remove it from the dataset.
#target_record = data.get_records([1])

epsilon = 100

n_ = 10000
n_targets = 100

n_shadow = 500
n_test = 30


def custom_metric(summary):
    # predictions = np.swapaxes(summary.predictions, 0, 1)
    scores = np.swapaxes(summary.scores, 0, 1)
    average_auc = 0
    # for label, prediction, score in zip(summary.labels, predictions, scores):
    for label, score in zip(summary.labels, scores):
        auc = .5 if len(np.unique(label)) < 2 else roc_auc_score(label, score)
        print(auc)
        average_auc += auc

    return average_auc / summary.labels.shape[0]



label = None
# filepath = "./tapas_test/snake_targets_wo_hhid"
# filepath = "./tapas_test/snake_target"
with open("./tapas_test/snake.json") as f:
    schema = json.load(f)
# target_records = _parse_csv(f"{filepath}.csv", schema, label or filepath)
target_records = data.sample(n_samples=n_targets)
data.drop_records(target_records.data.index, in_place=True)



# This step is important to ensure that the target record is not (incorrectly) added
# to training datasets that are not supposed to contain it.
#data_wo_target =  data.drop_records([1])
#data.drop_records([1],in_place=True)

# We attack the (trivial) Raw generator, which outputs its training dataset.
#generator = tapas.generators.Raw()
gen = 'PB_RQ_50'

# generator = tapas.generators.Raw()
generator = tapas.generators.ReprosynGenerator(PRIVBAYES, label="PB", epsilon=epsilon)
# generator = tapas.generators.ReprosynGenerator(PRIVBAYES(epsilon=epsilon, metadata="./tapas_test/snake.json",dataset="./tapas_test/snake.csv"), label="PB")
# generator = tapas.generators.ReprosynGenerator(MST(epsilon=epsilon, metadata="data/snake.json",dataset="data/snake.csv"), label="MST")
# generator = tapas.generators.ReprosynGenerator(MST, metadata="./tapas_test/snake.json", label="MST", epsilon=epsilon)
# generator = tapas.generators.ReprosynGenerator(MST, label="MST", epsilon=epsilon)


#generator = tapas.generators.ReprosynGenerator(MST(epsilon=1000.0,metadata="data/snake.json",dataset="data/snake.csv"), label="MST")
#
# We now define the threat model: what is assumed that the attacker knows.
# We first define what the attacker knows about the dataset. Here, we assume
# they have access to an auxiliary dataset from the same distribution.
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    # The attacker has access to 50% of the data as auxiliary information.
    # This information will be used to generate training datasets.
    data,
    auxiliary_split=0.5,
    #aux_data=data,snake_targets_wo_hhid.csv
    #test_data=target_record,
    # The attacker knows that the real dataset contains 5000 samples. This thus
    # reflects the attacker's knowledge about the real data.
    num_training_records=n_,
)

# We then define what the attacker knows about the synthetic data generator.
# This would typically be black-box knowledge, where they are able to run the
# (exact) SDG model on any dataset that they choose, but can only observe
# (input, output) pairs and not internal parameters.
sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator,
    # The attacker also specifies the size of the output dataset. In practice,
    # use the size of the published synthetic dataset.
    num_synthetic_records=n_,
)

# Now that we have defined the attacker's knowledge, we define their goal.
# We will here focus on a membership inference attack on a random record.
threat_model = tapas.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    # We here select the first record, arbitrarily.
    target_record=target_records,
    attacker_knowledge_generator=sdg_knowledge,
    # These are mostly technical questions. They inform how the attacker will
    # be trained, but are not impactful changes of the threat model.
    #  - do we generate pairs (D, D U {target}) to train the attack?
    generate_pairs=True,
    #  - do we append the target to the dataset, or replace a record by it?
    replace_target=True,
    # (Optional) nice display for training and testing.
    iterator_tracker=tqdm.tqdm,
)

# Next step: initialise an attacker. Here, we just apply the GroundHog attack
# with standard parameters (from Stadler et al., 2022).
attacker = tapas.attacks.GroundhogRQAttack(targets=target_records)
# attacker = tapas.attacks.GroundhogAttack()
#tapas.attacks.RandomTargetedQueryFeature(
#    target=target_records, order=15, number=575)

#

print("Training the attack...")
### Size of training data should be 100, 10 households, 5 are members
# Having defined all the objects that we need, we can train the attack.
attacker.train(
    # The TargetedMIA threat model is a TrainableThreatModel: it defines a method
    #  to generate training samples (synthetic_dataset, target_in_real_dataset).
    # This is why the threat model is passed to train the attacker.
    threat_model,
    # This is the number of training pairs generated by the threat model to
    # train the attacker.
    num_samples=n_shadow,
)


print("Testing the attack...")
# The attack is trained! Evaluate it within the test model.
# [explain why we split this way.]
attack_summary = threat_model.test(attacker, num_samples=n_test, ignore_memory=False)
# metrics = attack_summary.get_metrics()
average_auc = custom_metric(attack_summary)
print("Average auc:", average_auc)
# metrics.to_csv("result_"+gen+"_"+str(epsilon)+"_all.csv")

#report = tapas.report.MIAttackReport([attack_summary],metrics=['mia_advantage','auc'])
#report.load_summary_statistics([attack_summary])


'''
# Output nice, printable metrics that evaluate the attack.
report = tapas.report.MIAttackReport(attack_summary)
print(report)
#metrics = attack_summary.get_metrics()
#print("Results:\n", metrics.head())
'''

'''
# Publish human-readable graphs for this attack.
report = tapas.report.BinaryAIAttackReport(
    [attack_summary], metrics=["accuracy", "privacy_gain", "auc"]
)
report.publish("groundhog-snake")
'''