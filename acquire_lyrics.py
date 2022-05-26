import numpy as np
import pandas as pd
from env import api_token
import lyricsgenius as genius

api = genius.Genius(api_token, verbose=False)

def prep_songs():
    songs = pd.read_csv('charts.csv')
    songs['date'] = pd.to_datetime(songs.date)
    songs = songs.rename(columns={'song': 'title'})
    songs = pd.DataFrame(songs.groupby(['title', 'artist']).min().date).reset_index()
    return songs

def get_lyrics(title, artist):
    song = api.search_song(title, artist)
    lyrics = song.lyrics.replace('\n', ' ')
    # TODO handle verse and chorus tags
    return lyrics

for i in range(len(songs)):
    
    title = songs.iloc[[i]].title.values[0]
    artist = songs.iloc[[i]].artist.values[0]
    
    try:
        songs.loc[i, 'lyrics'] = get_lyrics(title, artist)
        songs.loc[i, 'status'] = 'lyrics acquired'
    except: 
        songs.loc[i, 'status'] = 'an error ocurred'

    songs.to_csv('songs.csv')
    
    print(f'{round((i / len(songs) * 100), 2)}%\t complete', end='\r')