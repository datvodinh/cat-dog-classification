import torch
import matplotlib.pyplot as plt
import seaborn as sns

def DataLoader(X_train,y_train,batch_size=64):
    """Split data into batch"""
    n = X_train.shape[0]
    idx = torch.randperm(n)
    X_new,y_new = X_train[idx],y_train[idx]
    for i in range(0,n,batch_size):
        begin,end = i, min(i+batch_size,n)
        yield X_new[begin:end],y_new[begin:end]

def FitModel(X_train,X_test,y_train,y_test,model,criterion,optimizer,epoch,batch_size):
    train_acc_list = []
    val_acc_list   = []
    for e in range(epoch):
        total_loss = 0
        for x,y in DataLoader(X_train,y_train,batch_size):
            y_pred = model.forward(x)
            loss = criterion(y_pred,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.detach().item()
            # print(loss.detach().item())
        total_loss /= int(X_train.shape[0] / batch_size)
        with torch.no_grad():
            y_train_pred = model.forward(X_train)
            y_test_pred  = model.forward(X_test)
            train_acc = torch.mean((torch.argmax(y_train_pred,dim=1)==torch.argmax(y_train,dim=1)) * 1.0)
            val_acc   = torch.mean((torch.argmax(y_test_pred,dim=1)==torch.argmax(y_test,dim=1)) * 1.0)
            train_acc_list.append(train_acc*100)
            val_acc_list.append(val_acc*100)
            print(f'EPOCH {e:>5} | LOSS: {total_loss:.4f} | TRAIN ACC: {train_acc* 100:.2f}% | TEST ACC: {val_acc*100:.2f}% |')
    
    Plot(train_acc_list,val_acc_list)

def Plot(train_acc_list,train_val_list):
    sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
    plt.rc('legend', fontsize=13)    # legend fontsize
    plt.rc('font', size=13)
    plt.figure(figsize=(12,6))
    plt.plot(train_acc_list)
    plt.plot(train_val_list)
    plt.legend(['Train','Test'])
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.show()