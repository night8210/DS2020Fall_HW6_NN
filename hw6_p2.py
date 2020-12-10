from p2.dataset import Dog_Cat_Dataset_Test, DataLoader
from p2.model import Model

import numpy as np 
import pandas as pd
import argparse, glob, torch
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./test')
parser.add_argument('--out_csv', type=str, default='./pred.csv')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def test():
    # load testset
    test_set = Dog_Cat_Dataset_Test(args.path)
    test_loader = DataLoader(test_set,batch_size=32)

    # load model
    model = Model()
    model.load_state_dict(torch.load('./p2/best_model.pth', 
                                     map_location='cpu')).to(device)
    model.eval()

    out_fnames,out_labels = [],[]
    for imgs, fnames in test_loader:
        predict = model(imgs.to(device))

        output = torch.argmax(predict.cpu(),dim=1)
        
        out_fnames += list(fnames)
        out_labels += list(output.tolist())

    df = pd.DataFrame(data={'image':pd.Series(out_fnames,dtype='str'),
                            'label':pd.Series(out_labels,dtype='int')})
    df.to_csv(args.out_csv, index=False)

if __name__=='__main__':
    test()

