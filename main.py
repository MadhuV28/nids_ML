#   Author        : *** Madhu B Vuyyuru ***
#   Last Modified : *** 12/5/2024 ***
#In the write up fodler, the roc curve for exp1 is showing
#delete it before final submission

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
import argparse
import helper
from sklearn import tree
from sklearn import metrics
from sklearn import neural_network
from sklearn import model_selection

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset, random_split
from torch import nn,optim

def exp1(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecure
    #split dataset into training and testing subsets, 80% for trainign and trest for testing
    X_train,X_test,Y_train,Y_test= train_test_split(data,labels,test_size=0.2,random_state=42)
    #only wrote hyperparametrs i played with
    model = tree.DecisionTreeClassifier(max_depth=3, 
                                        min_samples_split=100,
                                        min_samples_leaf=5,
                                        criterion="gini",
                                        class_weight="balanced",
                                        random_state=42,)
    #train decison tree 
    model.fit(X_train,Y_train)
    #predict clas labels for testing subset
    Y_prediction=model.predict(X_test)
    #probability for malicious in testing subset
    Y_probs=model.predict_proba(X_test)[:,1]

    #print classification report
    print("Classification report:")
    print(classification_report(Y_test,Y_prediction))

    #create confusion matrix cand comapre true labels with predicted labels
    cm=confusion_matrix(Y_test,Y_prediction)
    #confusion matrix plot
    plt.figure(figsize=(8,6))
    seaborn.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Benign','Malicious'],yticklabels=['Benign','Malicious'])
    plt.title("Confusion Martrix")
    plt.xlabel("Predicted")
    plt.ylabel("true")
    plt.savefig("confusion_matrix_exp1.png")
    plt.close()

    #decision tree plot
    #ask TA how to better visualize larger decision tree becuase exp2 tree is massive and I cannot see things
    plt.figure(figsize=(20,10))
    plot_tree(model,filled=True,feature_names=data.columns,class_names=['Benign','Malicious'])
    plt.title("Trained Decision Tree")
    plt.savefig("decision_tree_exp1.png")
    plt.close() 

    #fetch feautre impoortances found by deciison tre model
    feature_importances=model.feature_importances_

    #data frame to organize feautre importances, feautre is column names for data and importance is score for  each feature and sort
    importance_df=pd.DataFrame({
        'Feature': data.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance',ascending=False)
    
    #feauture importance plot.  should be a bar plot of feature importances 
    # x is scores fo importance on x-axis and y axis should be names of features   
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance',y='Feature',data=importance_df)
    plt.title("Feature Importance")
    plt.savefig("feature_importance_exp1.png")
    plt.close()

    #for ROC curve, we need false positve rate and true positvie trate and thresholds for decision making
    false_positive_rate,true_positive_rate, thresholds=roc_curve(Y_test,Y_probs)
    roc_auc=auc(false_positive_rate,true_positive_rate)
    #ROC curve plot
    plt.figure(figsize=(8,6))
    plt.plot(false_positive_rate,true_positive_rate,label=f"ROC CURVE(AUC= {roc_auc:.2f})")
    plt.plot([0,1],[0,1],linestyle="--",color="gray",label="Random Guess")
    plt.title("ROC CURVE")
    plt.xlabel("False Positve Rate")
    plt.ylabel("True Positve Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc_curve_exp1.png")

    #calculate optimal threshold
    #maximize difference between true positive rate and false positve rate and retrieve threshold that suits optimal index
    #coommented out after answering questions.  i dont want it messing with testers
    # optimal_index=np.argmax(true_positive_rate-false_positive_rate)
    # optimal_threshold=thresholds[optimal_index]
    # print(f"Optimal threshold is :{optimal_threshold:.2f}")

    """STUDENT CODE ABOVE"""
    return model


def exp2(data, labels):
    """STUDENT CODE BELOW"""
    #split dataset into training and testing subsets, 80% for trainign and trest for testing
    X_train,X_test,Y_train,Y_test= train_test_split(data,labels,test_size=0.2,random_state=42)
    #only wrote hyperparametrs i played with, the rest should be defualt 
    model = tree.DecisionTreeClassifier(max_depth=15, 
                                        min_samples_split=3,
                                        min_samples_leaf=2,
                                        criterion="gini",
                                        class_weight=None,
                                        random_state=42,)
    #train decison tree 
    model.fit(X_train,Y_train)
    #predict clas labels for testing subset
    Y_prediction=model.predict(X_test)

    #print classification report
    print("Classification report:")
    print(classification_report(Y_test,Y_prediction,target_names=['Benign','DoS','Probe','R2L','U2R']))
    #create confusion matrix cand comapre true labels with predicted labels
    cm=confusion_matrix(Y_test,Y_prediction)

    #confusion matrix plot
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['Benign','DoS','Probe','R2L','U2R'],
        yticklabels=['Benign','DoS','Probe','R2L','U2R']
    )
    plt.title("Confusion Martrix")
    plt.xlabel("Predicted")
    plt.ylabel("true")
    plt.savefig("confusion_matrix_exp2.png")
    plt.close()

    #trained decision tree
    #how can i make decision tree more visible.  rn the tree is so large i cannot see each node clearly
    plt.figure(figsize=(20,10))
    plot_tree(model,filled=True,feature_names=data.columns,class_names=['Benign','DoS','Probe','R2L','U2R'])
    plt.title("Trained Decision tree ")
    plt.savefig("decision_tree_exp2.png")
    plt.close()

    #fetch feautre impoortances found by deciison tre model
    feature_importances=model.feature_importances_

    #data frame to organize feautre importances, feautre is column names for data and importance is score for  each feature and sort
    importance_df=pd.DataFrame({
        'Feature': data.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance',ascending=False)
    #feauture importance plot.  should be a bar plot of feature importances 
    # x is scores fo importance on x-axis and y axis should be names of features  
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance',y='Feature',data=importance_df)
    plt.title("Feature Importance")
    plt.savefig("feature_importance_exp2.png")
    plt.close()
    # define model architecture
    """STUDENT CODE ABOVE"""
    return model

def exp3(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    #split dataset into training and testing subsets, 80% for trainign and trest for testing
    X_train,X_test,Y_train,Y_test= train_test_split(data,labels,test_size=0.2,random_state=42)
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,),
                                         activation="relu",
                                         solver="adam",
                                         alpha=.0001,
                                         max_iter=300,
                                         random_state=42)
    #train decison tree 
    model.fit(X_train,Y_train)
    #predict class labels for testing
    y_pred=model.predict(X_test)
    #probability for malicious in testing
    y_prob=model.predict_proba(X_test)[:,1]

    #print classification report
    print("Classifcaition report:")
    print(classification_report(Y_test,y_pred,target_names=["Benign","malicious"]))
    #create confusion matrix cand comapre true labels with predicted labels
    cm=confusion_matrix(Y_test,y_pred)
    #confusion matrix plto
    plt.figure(figsize=(10,8))
    seaborn.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Benign','Malicious'],yticklabels=['Benign','Malicious'])
    plt.title("Confusion Martrix")
    plt.xlabel("Predicted")
    plt.ylabel("true")
    plt.savefig("confusion_matrix_exp3.png")
    plt.close()

    #plot loss curve training epochs
    plt.figure(figsize=(10,6))
    plt.plot(model.loss_curve_,label="Loss Curve")
    plt.title("Loss CUrve Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("Loss_Curve_exp3.png")
    plt.close()

    #for ROC curve, we need false positve rate and true positvie trate and thresholds for decision making
    false_positive_rate,true_positive_rate, thresholds=roc_curve(Y_test,y_prob)
    roc_auc=auc(false_positive_rate,true_positive_rate)
    #ROC curve plot
    plt.figure(figsize=(8,6))
    plt.plot(false_positive_rate,true_positive_rate,label=f"ROC CURVE(AUC= {roc_auc:.2f})")
    plt.plot([0,1],[0,1],linestyle="--",color="gray",label="Random Guess")
    plt.title("ROC CURVE")
    plt.xlabel("False Positve Rate")
    plt.ylabel("True Positve Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc_curve_exp3.png")
    plt.show()
    #calculate optimal threshold
    #maximize difference between true positive rate and false positve rate and retrieve threshold that suits optimal index
    #coommented out after answering questions.  i dont want it messing with testers
    # optimal_index=np.argmax(true_positive_rate-false_positive_rate)
    # optimal_threshold=thresholds[optimal_index]
    # print(f"Optimal threshold is :{optimal_threshold:.2f}")
    """STUDENT CODE ABOVE"""
    return model

def exp4(data, labels):
    """STUDENT CODE BELOW"""
    # define model architecture
    #split dataset into training and testing subsets, 80% for trainign and trest for testing
    X_train,X_test,Y_train,Y_test= train_test_split(data,labels,test_size=0.2,random_state=42)
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,),
                                         activation="relu",
                                         solver="adam",
                                         alpha=.0001,
                                         max_iter=300,
                                         random_state=42)
    #train decison tree 
    model.fit(X_train,Y_train)
    #predict class labels for testing
    y_pred=model.predict(X_test)
    #probability for malicious in testing
    y_prob=model.predict_proba(X_test)
    target_names=["Benign","DoS","Probe","R2L","U2R"]

    #print classification report
    print("Classifcaition report:")
    print(classification_report(Y_test,y_pred,target_names=target_names))
    #create confusion matrix cand comapre true labels with predicted labels
    cm=confusion_matrix(Y_test,y_pred)
    #confusion matrix plto
    plt.figure(figsize=(10,8))
    seaborn.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=target_names,yticklabels=target_names)
    plt.title("Confusion Martrix")
    plt.xlabel("Predicted")
    plt.ylabel("true")
    plt.savefig("confusion_matrix_exp4.png")
    plt.close()

    #plot loss curve training epochs
    plt.figure(figsize=(10,6))
    plt.plot(model.loss_curve_,label="Loss Curve")
    plt.title("Loss CUrve Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("Loss_Curve_exp4.png")
    plt.close()
    """STUDENT CODE ABOVE"""
    return model


def exp5(data, labels):
    """STUDENT CODE BELOW"""
    # convert data to pytorch dataset
    dataset = helper.convert_to_pytorch_dataset(data, labels)
    #split dataset into the 80% training and 20% validation sets
    train_size=int(.8 * len(dataset))
    validation_size=len(dataset)-train_size
    train_dataset,validation_dataset=random_split(dataset,[train_size,validation_size])

    #data loaders
    batch_size=200
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    validation_loader=DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
    # define model architecture
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 5),
    )

    #optimizer and loss funciton
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=.001)

    #the training loop
    #times dataset passes thru model
    num_epochs=10
    for epoch in range(num_epochs):
        #model mode set tto trainign
        model.train()
        #init run loss for epoch
        run_loss=0.0
        #itterate over batches of training data
        for inputs, targets in train_loader:
            #clear gradients
            optimizer.zero_grad()
            #model predictions
            outputs=model(inputs)
            #loss predicitions vs actuals
            loss=criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            run_loss+=loss.item()
        average_loss=run_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.3f}")
        #validation
        #set to vevaluation mode
        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                outputs=model(inputs)
                _,predicted = torch.max(outputs.data,1)
                total+=targets.size(0)
                correct+=(predicted==targets).sum().item()
            accuracy=100 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")
        torch.save(model.state_dict(),"exp5_model.pt")
    """STUDENT CODE ABOVE"""
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, help="Experiment number")
    args = parser.parse_args()
    save_name = f"exp{args.exp}_model" + (".pt" if args.exp == 5 else ".pkl")
    if args.exp == 1:
        model = exp1(*helper.load_dataset(multiclass=False, normalize=False))
    elif args.exp == 2:
        model = exp2(*helper.load_dataset(multiclass=True, normalize=False))
    elif args.exp == 3:
        model = exp3(*helper.load_dataset(multiclass=False, normalize=True))
    elif args.exp == 4:
        model = exp4(*helper.load_dataset(multiclass=True, normalize=True))
    elif args.exp == 5:
        model = exp5(*helper.load_dataset(multiclass=True, normalize=True))
    else:
        print("Invalid experiment number")
    helper.save_model(model, save_name)
