
#count no. of times a song been played by diff. users
f=open("kaggle_users.txt",'r')
fp=open("kaggle_songs.txt",'r')
song_recommend=dict()

f1=open("D:\New folder\all\kaggle\kaggle_visible_evaluation_triplets.txt",'r')
user_collection=dict()

for line in f1:
    user,song,_=line.strip().split('\t')
    if user in user_collection:
        user_collection[user].add(song)
    else:
        user_collection[user]=set([song])


for user in f:
    user=user.strip()
    #fp.seek(0,0)
    for line in fp:
        song,_=line.strip().split(' ')
        if user in song_recommend:
            if song not in user_collection:
                song_recommend[user].add(song)
            else:
                continue
        else:
            song_recommend[user]=set([song])
      
f1.close()
fp.close()
f.close()

