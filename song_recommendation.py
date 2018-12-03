

import pandas
from sklearn.model_selection  import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation

#This step might take time to download data from external sources
triplets_file = '10000.txt'
songs_metadata_file = 'song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

#CREATE SUBSETOF DAATASET
song_df=song_df.head(10000)
#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

#CREATE A SONG RECOMMENDER
train_data,test_data=train_test_split(song_df,test_size=0.20,random_state=0)
#print(train_data.head(5))

#============================popularity based recommendation============================

pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')


#============================item_similarity based recommendation=================
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')
#==============================plot graph=============================
#Define what percentage of users to use for precision recall calculation
user_sample = 0.05

#Instantiate the precision_recall_calculator class <pm-popularity based recc. model>,<is_model-Item similarity based>
pr = Evaluation.precision_recall_calculator(test_data, train_data, pm, is_model)

#Call method to calculate precision and recall values
(pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)


#Code to plot precision recall curve
import pylab as pl

#Method to generate precision and recall curve
def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m2_recall_list, m2_label):
    pl.clf()
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    #pl.legend(loc="upper right")
    pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
    pl.show()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from tkinter import *
from PIL import ImageTk,Image
panel = Tk()
panel.title("musik")

#===============================IMAGES==================================
img=ImageTk.PhotoImage(Image.open('pic1.jpg'))
root=Label(panel,image=img,width=1600,height=800)
root.pack(side=TOP,fill=BOTH,expand=YES)



frame = Frame(root)
frame.pack()

Id= StringVar()
lis=StringVar()
art=StringVar()
def plot_data():
    plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")

def search_ar():
    sgsl = art.get()
    root3 = Tk()
    print(sgsl)
    root3.title("Musik")
    L1 = Label(root3, text="Artist greatest hits", font=(" Comic Sans MS", 15, "bold"), relief="sunken")
    L1.pack()

    # Show output of similer song list

    Lst4 = Listbox(root3, width=60, bg='gold', selectforeground='cyan')
    Lst4.pack(side=RIGHT, expand=0.5, fill=Y)

    arts=song_df[song_df['artist_name']== sgsl]
    #POPULAR ARTIST
    song_artist = arts.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
    similar=song_artist.sort_values(['listen_count'],ascending =False)
    #similar=similar['song']+"  -  " + similar['listen_count'].map(str)
    for q in similar['song']:
        Lst4.insert(END, q)
    root3.configure(bg='skyblue')
    root3.mainloop()



def search_s():
    sgsl = lis.get()
    root3 = Tk()
    root3.title("Musik")
    L1 = Label(root3, text="Similar Musics are here", font=(" Comic Sans MS", 15, "bold"), relief="sunken")
    L1.pack()

    # Show output of similer song list

    Lst4 = Listbox(root3, width=60, bg='gold', selectforeground='cyan')
    Lst4.pack(side=RIGHT, expand=0.5, fill=Y)
    sim=is_model.get_similar_items(sgsl)
    sim=sim['rank'].map(str)+"  -  " + sim['song']
    Lst4.insert(1,' '+sgsl)
    ct=2
    for x in sim:
        Lst4.insert(ct,x)
        ct+=1

    root3.configure(bg='skyblue')
    root3.mainloop()


def person():


    entry=Id.get()

    #Got The User Id in "entry" Perform ML

    #Show output From here
    #print(entry)
    root2 = Tk()
    root2.title("musik")

    b_frame = Frame(root2,bg="skyblue")
    b_frame.pack()

    #show User Id At Top
    L1 = Label(b_frame, text=" Result For User:-<<"+entry+">>", font=(" Comic Sans MS", 15, "bold", "underline"), relief="sunken")
    L1.pack()

    L2 = Label(b_frame, text="SONG playing History", font=(" Comic Sans MS", 15, "bold"), relief="sunken",width=40)
    L2.pack(side=LEFT,expand=1)

    L3 = Label(b_frame, text="Only for you", font=(" Comic Sans MS", 15, "bold"), relief="sunken",width=40)
    L3.pack(side=RIGHT,expand=1)

    L4 = Label(b_frame, text="Popular Songs", font=(" Comic Sans MS", 15, "bold"), relief="sunken", width=40)
    L4.pack(side=RIGHT, expand=1)

    #Output For perticuler User
    #There'll be 3 lists under "root2"
    #"Only For You","Populer Songs","played Songs"
    #enter items for PLAYING HISTORY
    Lst = Listbox(root2,width=60,bg='gold',selectforeground='cyan')
    Lst.pack(side=LEFT, fill=Y,expand=0.5)
    user_items = is_model.get_user_items(entry)
    cnt=1
    for x in user_items:
        Lst.insert(cnt,x)
        cnt+=1


    #enter items for ONLY FOR YOU
    Lst1 = Listbox(root2,width=60,bg='gold',selectforeground='cyan')
    Lst1.pack(side=RIGHT,expand=0.5, fill=Y)
    out=is_model.recommend(entry)
    out=out['rank'].map(str)+" - "+out['score'].map(str) + " - " + out['song']
    cnt=1
    for x in out:
        Lst1.insert(cnt,x)
        cnt+=1

    #enter items for POPULAR SONGS
    Lst2 = Listbox(root2,width=60,bg='gold',selectforeground='cyan')
    Lst2.pack(side=LEFT,fill=Y,expand=0.5)
    output=pm.recommend(entry)
    output=output['Rank'].map(str)+" - "+output['score'].map(str)+" - "+output['song']
    count=1
    for x in output:
        Lst2.insert(count,x)
        count+=1




    root2.geometry("1300x660+120+120")
    root2.configure(bg='skyblue')
    root2.mainloop()



Toolbar=Frame(root,bg="yellow green",relief="sunken")

L1 = Label(Toolbar, text="Search Songs", font=(" Comic Sans MS", 15, "italic"),fg='red', relief="sunken")
L1.pack(side=LEFT)

entry_2 = Entry(Toolbar, textvariable= lis, bd=4, font=("Times", 15, "bold"), bg='beige', fg='Brown')
entry_2.pack(side=LEFT)

button1 = Button(Toolbar, text="Search", font=("Times", 12, "bold"), bg='Blue',command=search_s)
button1.pack(side=LEFT)

entry_2 = Entry(Toolbar, textvariable=art, bd=4, font=("Times", 15, "bold"), bg='beige', fg='Brown')
entry_2.pack(side=LEFT)

button3 = Button(Toolbar, text="Artist Hits", font=("Times", 12, "bold"), bg='Blue',command=search_ar)
button3.pack(side=LEFT)

button = Button(Toolbar, text="Dig In",  font=("Times", 12, "bold"), bg= 'Blue' , command= person,padx=4,pady=4 )
button.pack(side=RIGHT)

entry_1=Entry(Toolbar,textvariable= Id, bd=5, font= ("Times", 15, "bold"),bg='beige',fg='Brown')
entry_1.pack(side=RIGHT)

L3 = Label(Toolbar, text="Enter UserID", font=(" Comic Sans MS", 15, "italic"),fg='red', relief="sunken",padx=4,pady=4)
L3.pack(side=RIGHT)

button2 = Button(Toolbar, text="Plot", font=("Times", 12, "bold"), bg='Blue',padx=2,pady=2,command=plot_data)
button2.pack()

Toolbar.pack(side=TOP,fill=X)

frame = Frame(root)
frame.pack()

Lb2 = Label(frame, text="All Songs", font=(" Comic Sans MS", 15, "bold"), relief="sunken", width=50)
Lb2.pack(expand=1)


#output of First Page
#All Songs and Hit Songs
List1= Listbox(root,width=100,bg='gold',selectforeground='cyan')
songlist=song_df['song']
for q in songlist:
    List1.insert(END,q)

List1.pack(side= LEFT,expand=0.5,fill=Y)


panel.geometry("1300x660+120+120")
panel.configure(bg='skyblue')
frame.configure(bg='#26261f')
panel.mainloop()
