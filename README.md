# Regression  Project

This project aims to **predict hotttnesss of a song**, it was implemented using a linear regression algorithmn. This is the **first part** of the complete project, which uses other techniques of machine learning

## Table of contents
* [Table of contents](#table-of-contents)
* [Installation](#installation)
* [Dataset description](#dataset-description)
  * [Analysis Group](#analysis-group)
  * [Metadata Group](#metadata-group)
  * [Musicbrainz Group](#musicbrainz-group)
* [References](#references)


## Installation
Setup your project folders and files

```bash
chmod +x install.sh
./install.sh
```

Look for music datasets
Understand datasets
Define scope for regression algorithm

[⬆️ Return](#table-of-contents)

## Dataset description

Dataset of 10k songs, random by MSD. It has three groups: metadata, analysis, musicbrainz

### Metadata Group
Metadata group with 20 attributes
```python
dtype([
    ('analyzer_version', 'S32'),        #0
    ('artist_7digitalid', '<i4'),       #1
    ('artist_familiarity', '<f8'),      #2 - Yes
    ('artist_hotttnesss', '<f8'),       #3 - Yes
    ('artist_id', 'S32'),               #4
    ('artist_latitude', '<f8'),         #5
    ('artist_location', 'S1024'),       #6
    ('artist_longitude', '<f8'),        #7
    ('artist_mbid', 'S40'),             #8
    ('artist_name', 'S1024'),           #9
    ('artist_playmeid', '<i4'),         #10
    ('genre', 'S1024'),                 #11 - Maybe
    ('idx_artist_terms', '<i4'),        #12
    ('idx_similar_artists', '<i4'),     #13
    ('release', 'S1024'),               #14
    ('release_7digitalid', '<i4'),      #15
    ('song_hotttnesss', '<f8'),         #16 - Target - Can be a NaN | Filter
    ('song_id', 'S32'),                 #17
    ('title', 'S1024'),                 #18
    ('track_7digitalid', '<i4')         #19
])
```
[⬆️ Return](#table-of-contents)

### Analysis Group
Analysis group with 31 attributes
```python
dtype([
    ('analysis_sample_rate', '<i4'),            #0
    ('audio_md5', 'S32'),                       #1
    ('danceability', '<f8'),                    #2 - X Empty [0, 1]
    ('duration', '<f8'),                        #3 - Yes
    ('end_of_fade_in', '<f8'),                  #4 - Yes
    ('energy', '<f8'),                          #5 - X Empty [0, 1]
    ('idx_bars_confidence', '<i4'),             #6 - X Empty
    ('idx_bars_start', '<i4'),                  #7 - X Empty
    ('idx_beats_confidence', '<i4'),            #8 - X Empty
    ('idx_beats_start', '<i4'),                 #9 - X Empty
    ('idx_sections_confidence', '<i4'),         #10 - X Empty
    ('idx_sections_start', '<i4'),              #11 - X Empty
    ('idx_segments_confidence', '<i4'),         #12 - X Empty
    ('idx_segments_loudness_max', '<i4'),       #13 - X Empty
    ('idx_segments_loudness_max_time', '<i4'),  #14 - X Empty
    ('idx_segments_loudness_start', '<i4'),     #15 - X Empty
    ('idx_segments_pitches', '<i4'),            #16 - X Empty
    ('idx_segments_start', '<i4'),              #17 - X Empty
    ('idx_segments_timbre', '<i4'),             #18 - X Empty
    ('idx_tatums_confidence', '<i4'),           #19 - X Empty
    ('idx_tatums_start', '<i4'),                #20 - X Empty
    ('key', '<i4'),                             #21
    ('key_confidence', '<f8'),                  #22
    ('loudness', '<f8'),                        #23
    ('mode', '<i4'),                            #24
    ('mode_confidence', '<f8'),                 #25
    ('start_of_fade_out', '<f8'),               #26
    ('tempo', '<f8'),                           #27
    ('time_signature', '<i4'),                  #28
    ('time_signature_confidence', '<f8'),       #29
    ('track_id', 'S32')                         #30
])
```

[⬆️ Return](#table-of-contents)
### Musicbrainz Group
```python
dtype([
    ('idx_artist_mbtags', '<i4'),   #0
    ('year', '<i4')                 #1
])
```

Some values are undefined, before the training part, I should ensure we have only have data with valid values. The 10k records downloaded from MSD, are not provided with danceability and energy. In consecuense, the objective was reformulated.

## Linear regression

Using 14 attributes(including bias), with a learning rate of 10, it diverges. Parameters oscilate between positive and negative values, but they can not converge

[![asciicast](https://asciinema.org/a/GMkg4hLF29ZpCEEFJvMMrJ0cW.svg)](https://asciinema.org/a/GMkg4hLF29ZpCEEFJvMMrJ0cW)


We can get better results with a lower learning rate 0.01

[![asciicast](https://asciinema.org/a/Ml9dJZwIKHalvsGOjlinwXqnn.svg)](https://asciinema.org/a/Ml9dJZwIKHalvsGOjlinwXqnn)

[⬆️ Return](#table-of-contents)
## References
* https://github.com/mdeff/fma
* http://millionsongdataset.com/musixmatch/
* https://medium.com/atchai/in-search-of-the-perfect-music-dataset-ed7e111d3b7e

Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.

[⬆️ Return](#table-of-contents)

## Extra
```bash
#/mnt/d/tmp_ml/msd/MillionSongSubset/data
find . -type f -name \*.h5 -exec cp \{\} /mnt/d/tmp_ml/msd/MillionSongSubset/tracks/ \;
```

