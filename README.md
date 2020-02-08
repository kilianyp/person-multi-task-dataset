# Description
A multi-task/multi-dataset training pytorch framework.

Framework was built for "[Visual Person Understanding through Multi-Task and Multi-Dataset Learning](https://arxiv.org/pdf/1906.03019.pdf)".

It can train the following tasks:
- Person ReIdentification
- Person Pose Estimation
- Person Body Parts Segmentation
- Person Attribute Learning
- Person Classification

On the following datasets:
- Market-1501
- Duke MTMC
- LIP
- MPII

The following papers are (partially) included: 
- "In Defense of the Triplet Loss for Person Re-Identification" [Link](https://arxiv.org/abs/1703.07737).
- "Learning Discriminative Features with Multiple Granularities for Person Re-Identification" [Link](https://arxiv.org/abs/1804.01438)
- "Multi-Task Learning as Multi-Objective Optimization" [Link](http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization)

Videos are from training sequences of the [MOT17](https://motchallenge.net/data/MOT17/) challenge:
- [02](https://drive.google.com/open?id=1nq1KLH0X5j26YaLCjQOFQ8x_KydonVds)
- [05](https://drive.google.com/open?id=1E22xH5TWhmKyj_kosb-r-16WQcWJkzCW)
- [09](https://drive.google.com/open?id=1TNsgqN4QWHw2-m6TmgtboAi4Lw_RPTCL)
- [10](https://drive.google.com/open?id=1J2mzWhUu6EcotEqW_C9OlXmFkZoUTDo-)
- [11](https://drive.google.com/open?id=1iysZ0LHzagVynx6etS8SOkZsxPoZ44b1)

Shown model used ResNet-50 backbone. On a 1080 GTX, ~120 cropped images per second could be processed.

NOTE: Ground truth bounding boxes were used. Bounding box color indicate gender. 

# Usage
***Code needs clean-up and is provided as is. If there's a general interest, I will look into making things cleaner***

# Requirements
- numpy
- Pillow
- h5py
- scipy
- torch
- torchvision
- sacred
- imgaug

The GroupNorm implementation stems from [here](https://github.com/chengyangfu/pytorch-groupnormalization).

# Installation
pip install -r requirements.txt (for training)

To be able to use omniboard:
- Install mongodb
- Install npm
- Install omniboard

### For no-admin rights user:
#### mongodb
Install tarball from https://docs.mongodb.com/v3.2/tutorial/install-mongodb-on-linux/

#### npm
I recommend setting npm up in such way that installed modules can be run from command line.

Therefore:
- Change the npm install directory:
-- create a file called .npmrc with the content
`prefix=${HOME}/.npm-packages`
-- add the following to your .bashrc
`# Node setup for global packages without sudo
NPM_PACKAGES="${HOME}/.npm-packages"
NODE_PATH="$NPM_PACKAGES/lib/node_modules:$NODE_PATH"
PATH="$NPM_PACKAGES/bin:$PATH"`


#### omniboard
`npm install omniboard`

# Starting 
mongod --dbpath mongo
## Access database from remote host
- Create a config file with:
`bind_ip = 127.0.0.1, ip1, ip2`
where ip1 and ip2 are the assigned apis within the network the database should be accessible from.

Then start mongodb with
mongod --dbpath mongo -f mongod.conf

omniboard -m host:port:db
### Password Protect Database
- Create admin user.
```
use admin
db.createUser(
  {
    user: "myUserAdmin",
    pwd: "abc123",
    roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
  }
)
```
- restart database with `mongod --auth` + additional parameters
#### Additionally
This is not really necessary but for one good practice but also seems to be necessary to work 
correctly with omniboard
- Create another user in your experiment database
```
use master
db.createUser(
  {
    user: "myUser",
    pwd: "abc123",
    roles: [ { role: "readWrite", db: "master" } ]
  }
)
```
For sacred, it is possible to use the admin User, but it is more secure to use a specific, more limited user.
```
MONGO_USER = "myUser"
MONGO_PW = "abc123"
DB_NAME = "master"
```

## Starting Omniboard
### without password
`omniboard -m host:27017:master`
### with password
For omniboard, I could not get it to run using the admin user (it would always connect to the admin database).
`omniboard --mu "mongodb://myUser:abc123@localhost/master?authMechanism=SCRAM-SHA-1" master`
--authSource=admin
No need to create another user

## Connecting from cluster
I recommend using ngrok.
- `ngrok tcp 27017` to punsh a tunnel to your mongo database. Ngrok will give you an URL you can connect to.
- Use this URL as your host.




# Train
```
python3 main.py with configs.json
```

For market, you can find them [here](https://github.com/VisualComputingInstitute/triplet-reid/tree/master/data):


# Evaluation

You can use embed.py to write out embeddings that are compatible with the 
evaluation script.

```
python3 main.py evaluate_from_confipython3 main.py evaluate_experiment with evaluate.json evaluation.experiment=/dir/to/experiment
```
To calculate the final scores, please use the evaluation script from 
[here](https://github.com/VisualComputingInstitute/triplet-reid#evaluating-embeddings)!

# Scores without Re-rank (and pretrained models) 
### Market-1501
#### Trinet
Settings: 
- P=18 
- K=4
- dim=128

Download Model ([GoogleDrive](https://drive.google.com/open?id=1eNJuLxRz3dJ0MkVjoLP6vshxZUn_NLn0))

|Test time augmentation| mAP | top-1 | top-5| top-10|
|---|---:|---:|---:|---:|
| None | 65.06% | 80.31% | 92.25% | 94.71% |
| With TenCrop |  69.44% | 83.40% | 93.59% | 96.17% |


#### MGN

Settings:


| Test time augmentation | mAP | top-1 | top-5| top-10|
|---|---:|---:|---:|---:|
| With Horizontal Flip | 83.17% | 93.62% | 97.86% | 98.66% |

# Citing
If you used this project, please consider citing
```
@inproceedings{pfeiffer2019visual,
  title={Visual Person Understanding Through Multi-task and Multi-dataset Learning},
  author={Pfeiffer, Kilian and Hermans, Alexander and S{\'a}r{\'a}ndi, Istv{\'a}n and Weber, Mark and Leibe, Bastian},
  booktitle={German Conference on Pattern Recognition},
  pages={551--566},
  year={2019},
  organization={Springer}
}
```

