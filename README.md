This folder contains the data, code, results, and write-up for the dissertation chapter examining r/ChangeMyView by Ryan Funkhouser. This project aims to test the influence of small stories on the outcomes of deliberations on r/ChangeMyView. To accomplish this, it will train a supervised machine learning text classification algorithm to identify which comments contain small stories and test the relationship between persuasive outcomes (receiving a 'delta' in the subreddit) and the presence or absence of small stories. 

The code starts with a subfolder of "One Time" code to get the deltalog IDs of which comments received deltas. Since I've already run it, I kept it separate since rerunning the rest of the code does not require this. Similarly, it includes the code to extract the r/ChangeMyView comments from zst to JSON. The pipline of the actual code to process the data is labeled in stages from 1-5. 

Link to the raw original data file for all r/ChangeMyView comments can be found here: https://purdue0-my.sharepoint.com/:u:/g/personal/funkhour_purdue_edu/EdBy4nGkGPhAriMEhld-jDQBId3oWRq-MZ9d3T-V9vjZPQ

Link to CSV of IDs for all comments that received deltas from the OP: https://purdue0-my.sharepoint.com/:x:/g/personal/funkhour_purdue_edu/ES2n_XxlHBNOqc_OdPA-f3IBirw25G4LJBPwZefggdW3uA?e=OFbZ9V 