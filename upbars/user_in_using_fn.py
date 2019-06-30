import numpy as np
import keras
import pandas as pd
import numpy as np

from collections import OrderedDict

age = 0
data_folder = './data'
###retrive id of user liked application
def rec_user(id):

    records = pd.read_csv(data_folder + '/user'+id+'.csv')

    #uid = records.uid[:1]
    app = records.app
    userliked = []
    #records.drop(columns='index',inplace=True)
    #print(uid)
    #print(app)
    a = list(app)
    #print(a[0])
    record = pd.read_csv(data_folder + '/database.csv')
    # db = record.to_dict()
    mydb = dict(zip(record.id, record.App))
    dbkey = [*mydb]
    dbvalues =list(mydb.values())
    for x in range(len(a)):
        for val in range(len(dbvalues)):
            if dbvalues[val].find(a[x]) == 0 :
                #print("found id")
                userliked.append(dbkey[val])

    #print(userliked)
    return userliked

userliked = rec_user("2")
print("userliked application :" ,userliked)
#### find simlilar users
def cfuser_gen(u):
    #print(u)
    l = []
    can = pd.read_csv(data_folder + '/candidate.csv')
    userid = list(can.userid)
    #print(userid)
    appid = list(can.id)
    #print(appid)
    for i in range(len(u)):
        for x in range(len(appid)):
            if u[i] == appid[x]:
                l.append(userid[x])

    return l

cf_user =cfuser_gen(userliked)
####redundant id removeinig
#print(cf_user)
x = np.array(cf_user)
cf_user=list(np.unique(x))
print("clustered users :",cf_user)
#### obtaining the candidate set
def candidate_set(cf):
    l=[]
    can = pd.read_csv(data_folder + '/candidate.csv')
    user = list(can.userid)
    app = list(can.id)
    for i in range(len(cf)):
        for x in range(len(user)):
            if cf[i]==user[x]:
                l.append(app[x])

    return l


#redundat id removal from candidate set
can_set = candidate_set(cf_user)

#avoid redudantant
y=np.array(can_set)
can_set=list(np.unique(y))

print("size of candidate :", len(can_set))

#substract userliked from candidate
rec_candidate = [x for x in can_set if x not in userliked]
print("size of recommend candidate :", len(rec_candidate))
print("candiate :",rec_candidate)

list = []
#### user contextual preference
def user(id) :
    print("hello user \n your id is" + id)
    ## piad free
    print("intrested in paid app enter 1 ")
    paid= int(input())
    if paid==1 :
        print("preferred paid")
        list.append(1)
    else :
        print("prefered free")
        list.append(0)

    #android version
    print("enter your android version")
    av = input()
    if av == '1.5':
        list.append(0)
    elif av=='2.1':
        list.append(1)
    elif av=='2.2':
        list.append(2)
    elif av == '2.3':
        list.append(3)
    elif av=='2.3.3':
        list.append(4)
    elif av=='3.0':
        list.append(5)
    elif av=='3.2':
        list.append(6)
    elif av=='4.0':
        list.append(7)
    elif av=='4.0.3':
        list.append(8)
    elif av=='4.1':
        list.append(9)
    elif av=='4.2':
        list.append(10)
    elif av=='4.3':
        list.append(11)
    elif av=='4.4':
        list.append(12)
    elif av=='5':
        list.append(13)
    elif av=='6':
        list.append(14)
    else :
        list.append(7)

    #inpp  purchase

    print("inap prference intrested enter 1")
    inapp= int(input())
    if inapp==1 :
        print("inapp preffered")
        list.append(1)
    else :
        print("not intrested to inapp")
        list.append(0)
    ## privacy
    print("privacy preference/n")
    print("location acess enter 1 ")
    loc = int(input())
    if loc ==1 :
        print("location acess allowed")
        list.append(0)
    else :
        print(" location acess not allowed")
        list.append(0)



    print("contact read permission")
    cr = int(input())
    if cr == 1:
        print("contact read allowed")
        list.append(1)
    else:
        print("contact read not allowed")
        list.append(0)


    print("contact modification enter 1")
    cw = int(input())
    if cr == 1:
        print("contact modification allowed")
        list.append(1)
    else:
        print("contact modification allowed")
        list.append(0)


    print("media read permission enter 1 ")
    mr = int(input())
    if mr == 1:
        print("media read allowed")
        list.append(1)
    else:
        print("mediaread not allowed")
        list.append(0)


    print("media write permission enter 1 ")
    mw = int(input())
    if mw == 1:
        print("media write allowed")
        list.append(1)
    else:
        print("media write not allowed")
        list.append(0)

    print("storage read permission enter 1 ")
    sr = int(input())
    if sr == 1:
        print("storage read allowed")
        list.append(1)
    else:
        print("storage read not allowed")
        list.append(0)


    print("storage write permission enter 1 ")
    stw = int(input())
    if stw == 1:
        print("storage write allowed")
        list.append(1)
    else:
        print("storage write not allowed")
        list.append(0)


    print("camera acess 1 ")
    cm = int(input())
    if cm == 1:
        print("camera acess allowed")
        list.append(1)
    else:
        print("camera acess not allowed")
        list.append(0)

    print("intrested in ad contents enter 1")
    add = int(input())
    if add ==1:
        print("add content intrested")
        list.append(1)
    else:
        print("add not preferred")
        list.append(0)
    #label


    print("size prefrence : 1 for below 20 , 2 for below 50 ,3 below 100")
    size = int(input())
    if size == 1:
        print("prefered below 20")
        list.append(1)
        list.append(0)
        list.append(0)
    elif size== 2:
        print("prefered below 50")
        list.append(0)
        list.append(1)
        list.append(1)
    else :
        print("prefered 100 above")
        list.append(0)
        list.append(0)
        list.append(1)
    ### category
    print("category preference")
    print("1:art and design")
    print("2:books and reference")
    print("3:business")
    print("4:communication")
    print("5:educaion")
    print("6:entertainment")
    print("7:game")
    cat = int(input())
    if cat == 1:
        print("art and design prefered")
        list.append(1)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
    elif cat == 2:
        print("books and refernce prefered")
        list.append(0)
        list.append(1)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
    elif cat == 3:
        print("business prefered")
        list.append(0)
        list.append(0)
        list.append(1)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
    elif cat == 4:
        print("communication prefered")
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(1)
        list.append(0)
        list.append(0)
        list.append(0)
    elif cat == 5:
        print("education")
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(1)
        list.append(0)
        list.append(0)
    elif cat == 6:
        print("entertainment prefered")
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(1)
        list.append(0)
    else :
        print("game prefered")
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(0)
        list.append(1)
    print("rating prefernce ")
    print("1:above 3")
    print("2:above 4")
    print("any number for no prefernce")
    rating = int(input())
    if rating == 1:
        print("above 3 preferred")
        list.append(1)
        list.append(1)
        list.append(0)
    elif rating == 2 :
        print("above 4 preferred")
        list.append(0)
        list.append(1)
        list.append(0)
    else :
        print("no rating prefernce")
        list.append(0)
        list.append(0)
        list.append(0)

    print("enter your age")
    age=int(input())
    if age<=17:
        list.append(1)
        list.append(1)
        list.append(0)
        list.append(1)
    elif age <= 10:
        list.append(1)
        list.append(0)
        list.append(0)
        list.append(0)
    else:
        list.append(1)
        list.append(1)
        list.append(1)
        list.append(1)
    return(list)



a = user('2')
a=np.array(list)
##reshaoing contextual features
ap= a.reshape(1,29)
#m= keras.models.load_model('models.h5')
#####loading saved model
m=keras.models.load_model('model.h5')
data_folder ='./data'
record = pd.read_csv(data_folder + '/database.csv')
#db = record.to_dict()
mydb = dict(zip(record.id, record.App))
#for x, y in mydb.items():
 #   print(x,y)
#ap= a.reshape(1,28)
#list = [191,118,114,102,300]

#####demographi sparse input consider age

if not rec_candidate:
    print ("enter age group")
    liste=[]
    age = int(input())
    if age <= 17 and age > 10:
        liste.append(0)
        liste.append(0)
        liste.append(0)
        liste.append(1)
    elif age <= 10:
        liste.append(0)
        liste.append(1)
        liste.append(0)
        liste.append(0)
    else:
        liste.append(1)
        liste.append(0)
        liste.append(0)
        liste.append(0)

    l = []
    j = 0
    record = pd.read_csv(data_folder + '/agecan.csv')
    # print(record.head())
    e = record.Everyone
    f = record.everyoneabove10
    g = record.Matureabove17
    h = record.Teen
    id = record.id
    for i in range(len(f)):
        # print("list is ", list)
        if liste[0] == e[i] and liste[1] == f[i] and liste[2] == g[i] and liste[3] == h[i]:
            # print("list is ",list)
            l.append(id[i])
    print("list is ", l)
    rec_candidate = np.unique(l)
    print("demography based recomend candidate:", rec_candidate)


#print(m.predict([pd.Series(id),ap]))
###store id and predicted value in dictionary for whole candidate
dict={}
for i in range(len(rec_candidate)):
    key = rec_candidate[i]
    value =  m.predict([pd.Series(rec_candidate[i]), ap])

    dict[key] = value
for x, y in dict.items():
    print(x,y)

####sorting the candidate and recommend top 10
#for x,y in dict.items():
#   print(x,y)

dd = OrderedDict(sorted(dict.items(), key=lambda x: x[1], reverse=True)[:10])
print("recommended")
for x, y in dd.items():
    print(x,y)
#recomended apps id ie:key of dd
a = [*dd]
#print(a)
#retreving names of the app from DB
for i in range(len(a)):
    print(str(i)+")"+ mydb[a[i]])
