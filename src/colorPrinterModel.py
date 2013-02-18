'''
Created on 18.02.2013
    This module is for functions to deal with printer stuffs: printer model, forward printer model
    and inverse printer model a.k.a. 
@author: gerjer
'''

import math
import numpy as np

def Demichel3Colorants(colorantCombination):
    #weights = np.zeros(math.pow(2,np.shape(colorantCombination)[0]),np.shape(colorantCombination)[1])
    weights = np.zeros((8,np.shape(colorantCombination)[1]))
    for i in np.arange(0,np.shape(colorantCombination)[1]):
        weights[0,i] = (1-colorantCombination[0,i]) * (1-colorantCombination[1,i]) * (1-colorantCombination[2,i])
        weights[1,i] = colorantCombination[0,i]     * (1-colorantCombination[1,i]) * (1-colorantCombination[2,i])
        weights[2,i] = (1-colorantCombination[0,i]) * colorantCombination[1,i]     * (1-colorantCombination[2,i])
        weights[3,i] = (1-colorantCombination[0,i]) * (1-colorantCombination[1,i]) * colorantCombination[2,i]
        weights[4,i] = colorantCombination[0,i]     * colorantCombination[1,i]     * (1-colorantCombination[2,i])
        weights[5,i] = colorantCombination[0,i]     * (1-colorantCombination[1,i]) * colorantCombination[2,i]
        weights[6,i] = (1-colorantCombination[0,i]) * colorantCombination[1,i]     * colorantCombination[2,i]
        weights[7,i] = colorantCombination[0,i]     * colorantCombination[1,i]     * colorantCombination[2,i]    
    return weights
    
    
'''
def Demichel(colorant):
    ''
    The function compute the Demichel weights for a list of colorant combination.
    The data "colorant" is of shape n x m where n is the number of colorant and m the 
    number of combination.
    We know that for n = 3 we have 2^n = 2^3 = 8 weights, one for each Neugebauer Primary.
    ''
    print np.shape(colorant)
    if np.shape(colorant)[0] >7:
        print 'it is way to big.'
        weigthsDemichel  = 0
        return weigthsDemichel 
    else:
        weigthsDemichel = np.zeros((8,np.shape(colorant)[0]))
        print weigthsDemichel
        print np.shape(weigthsDemichel)
    
    return weigthsDemichel

%function weigth = Demichel2(colorants)
function weights = Demichel2(colorants)

ss      = size(colorants);
weigths = zeros(2^ss(1),ss(2));

for ii=1:ss(2)
    weights(:,ii)=FunDemichel(colorants(:,ii));
end

%********NESTED FUNCTION**********
function ww=FunDemichel(Col)
L=length(Col);
area=zeros(1,L);
for i=1:L
    config=nchoosek(1:L,i);
    [li co]=size(config);
    temp=zeros(li,L);
    for j=1:li
        temp(j,config(j,:))=1;
    end
    area=[area; temp];
end
non_area = ~area;
tab_with_dot = (diag(Col)*ones(L,2^L))';
tab_without_dot = ones(2^L,L)-tab_with_dot;
ww = prod((tab_with_dot .* area + tab_without_dot.*non_area)');
clear area temp
'''