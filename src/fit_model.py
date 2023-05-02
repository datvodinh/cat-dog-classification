import torch

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
            print(f'EPOCH {e:>5} | LOSS: {total_loss:.4f} | TRAIN ACC: {train_acc* 100:.2f}% | VAL ACC: {val_acc*100:.2f}% |')
    
    Plot(train_acc_list,val_acc_list)

def Plot(train_acc_list,train_val_list):
    pass