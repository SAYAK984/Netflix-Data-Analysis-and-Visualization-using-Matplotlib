import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1
# Fetching the csv file......
df = pd.read_csv("I:/ML_FOLDER/MATPLOTLIB_FOLDER/Dataset_Folder/netflix_titles.csv")

#2
# Dataset preview.....
print(f"First 5 rows of the DataFrame.....\n{df.head()}\n")
print(f"Similarly, last 5 rows of the DataFrame.....\n{df.tail()}\n")

#3
# Understanding the Datasets......

# Checking total rows and columns of the dataset.......
print(f"Structure of the DataFrame......")
print(f"Entry rows = {df.shape[0]}")
print(f"Total columns = {df.shape[1]}\n")

# Getting informations about all columns.......
print("Information about the columns......")
df.info()
"""
Information about all the columns of the DataFrame.......
#   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   show_id       8807 non-null   str  
 1   type          8807 non-null   str  
 2   title         8807 non-null   str  
 3   director      6173 non-null   str  
 4   cast          7982 non-null   str  
 5   country       7976 non-null   str  
 6   date_added    8797 non-null   str  
 7   release_year  8807 non-null   int64
 8   rating        8803 non-null   str  
 9   duration      8804 non-null   str  
 10  listed_in     8807 non-null   str  
 11  description   8807 non-null   str
"""

# Getting statistical-summarization of all the numeric-attributes.......
print(f"\nGetting the statistical summarization of the numeric-attributes.....\n{df.describe()}\n")


#4
# Handling the missing values.......

# counting the NaN/null-values in each columns (if any)......
print(f"Counting the NaNs in each columns......\n{df.isna().sum()}\n")
"""
Counting the NaNs in each columns......
show_id            0
type               0
title              0
director        2634 <-
cast             825 <-
country          831 <-
date_added        10 <-
release_year       0
rating             4 <-
duration           3 <--
listed_in          0
description        0
dtype: int64
"""


# counting the infinity values in each columns (if any)......
print(f"Counting the infinity values in each columns.......\n{df.isin([np.inf, -np.inf]).sum()}\n")


# Removing the NaN/Null-values row-wise by checking only these following specific columns.......
# 'type', 'release_year', 'rating', 'country', 'duration'
# NaN/Null-values of other attributes are ignored...... 
#df.dropna(axis = 0, inplace = True)
df.dropna(subset = ['type', 'release_year', 'rating', 'country', 'duration'], inplace = True)


#5
# Data visualization for the Netflix Datasets........
# Plotting multiple graphs (plots, bar, pie, histogram, scatter) in the single window......

fig, axes = plt.subplots(2,2, figsize = (10,7))
fig.suptitle("Summarization for Netflix Movies and TV shows (Part-1)")

"""
"""
#5.1
# Data-visualization for the Show-type comparisions (Movie Vs TV-shows)....... 
# Checking total no. of Movies and TV-shows (bar-chart req.d)......
print(df["type"].value_counts())
show_type = df["type"].value_counts()
print(show_type.index)
print(show_type.values,"\n")

# Plotting bar-chart for comparision between Movies and TV-shows.......
axes[0, 0].bar(show_type.index, show_type.values, color = ["orange", "yellow"], edgecolor = "black", label = ["Total no. of Movies", "Total no. of TV Shows"])

axes[0, 0].set_title("Netflix Movies Vs TV Shows (Total count)")
axes[0, 0].set_xlabel("Show Type ------>")
axes[0, 0].set_ylabel("Total count ------>")

axes[0, 0].legend(loc = "upper right", fontsize = 9)

axes[0, 0].grid(color = "gray", linestyle = ":", linewidth = 0.50)


#5.2
# Data-visualization for the Content-Ratings Proportions (Pie-chart req.d).......
# Checking the Content-Ratings for both Movies and TV-shows......
print(f"\nVerification for the 'rating'-attribute for NaNs (if any) = {df['rating'].isna().sum()}")

print(df["rating"].value_counts())
rating_type = df["rating"].value_counts()
print(rating_type.index)
print(rating_type.values,"\n")


# Plotting Pie-chart for the Content-Ratings Proportion.......
axes[0, 1].pie(rating_type.values, labels = rating_type.index, autopct = "%1.1f%%")

axes[0, 1].set_title("Rating-type proportion of Movies and TV Shows")



#5.3
# Data-visualization for Distribution (Histogram req.d).......
# Checking the Distribution w.r.t Duration for both Movies and TV-shows......
print(f"\nVerification for the 'duration'-attribute for NaNs (if any) = {df['duration'].isna().sum()}")


# For Movies.......................................
df_movies = df[df["type"] == "Movie"]  # Separate DataFrame for movies......

print(df_movies["duration"].value_counts())
duration_movie_type = df_movies["duration"].value_counts()
print(duration_movie_type.index)
print(duration_movie_type.values,"\n")

print(f"df['duration'] for movies is......\n{df_movies['duration']}\n")

time_series = df_movies["duration"].str.replace(" min", "").astype(int)
print(f"Time list for movies is.......\n{time_series}")

"""
Time list is.......
0        90
7       125
9       104
12      127
24      166
       ... 
8801     96
8802    158
8804     88
8805     88
8806    111
Name: duration, Length: 5687, dtype: int64
"""
# Plotting Histogram for Distribution of Movies w.r.t its durations.......
axes[1, 0].hist(time_series.values, bins = 30, color = "yellow", edgecolor = "black", label = "Total no. of Movies at this duration range")

axes[1, 0].set_title("Distribution of movie duration")
axes[1, 0].set_xlabel("Range of durations (in minutes) -------->")
axes[1, 0].set_ylabel("Total no. of Movies --------->")

axes[1, 0].legend(loc = "upper right", fontsize = 8)

axes[1, 0].grid(color = "gray", linestyle = ":", linewidth = 0.50)
#......................................................


# For TV-Shows.........................................
df_tv_show = df[df["type"] == "TV Show"]
print(df_tv_show["duration"].value_counts())
duration_tvshow_type = df_tv_show["duration"].value_counts()
print(duration_tvshow_type.index)
print(duration_tvshow_type.values,"\n")

print(f"df['duration'] for TV-show is......\n{df_tv_show['duration']}\n")

time_series_2 = df_tv_show["duration"].str.replace(" Seasons", "")
time_series_2 = time_series_2.str.replace(" Season", "").astype(int)
print(f"time series for TV-show is......\n{time_series_2}\n")


# Plotting Histogram for Distribution of TV-shows w.r.t its Seasons.......
axes[1, 1].hist(time_series_2.values, bins = 30, color = "orange", edgecolor = "black", label = "Total no. of TV shows at this duration range")

axes[1, 1].set_title("Distribution of TV-show Seasons")
axes[1, 1].set_xlabel("Range of Seasons --------->")
axes[1, 1].set_ylabel("No. of TV shows -------->")

axes[1, 1].legend(loc = "upper right", fontsize = 10)

axes[1, 1].grid(color = "gray", linestyle = ":", linewidth = 0.50)
#.......................................................

#print(f"TV show with 15 seasons.....\n{df_tv_show.loc[df['duration'] == '15 Seasons', ['title', 'director']]}\n")

plt.tight_layout()
plt.savefig("I:/ML_FOLDER/MATPLOTLIB_FOLDER/MatplotLib_Project/Netflix_Plot_3.jpg", dpi = 350, bbox_inches = "tight")


"""
"""
#5.4
# Data-Visualization for the Total Movies/TV-shows released Vs Releasing-Year (Scatter-plot req.d).....

fig_2, axes_2 = plt.subplots(1,2, figsize = (10, 7))
fig_2.suptitle("Summarization for Netflix Movies and TV shows (Part-2)")


# For movies......
year_list = df_movies["release_year"].value_counts()
print(year_list)
print(year_list.index)
print(year_list.values,"\n")

axes_2[0].scatter(year_list.index, year_list.values, color = "red", marker = "o", label = "Total movie released in a year")
axes_2[0].set_title("Movies Released Vs Releasing-Year")
axes_2[0].set_xlabel("Releasing-Year ------->")
axes_2[0].set_ylabel("Total Movies released --------->")

axes_2[0].legend(loc = "upper left", fontsize = 9)

axes_2[0].grid(color = "gray", linestyle = ":", linewidth = 0.50)


# For TV-Shows......
year_list_2 = df_tv_show["release_year"].value_counts()
print(year_list_2)
print(year_list_2.index)
print(year_list_2.values,"\n")

axes_2[1].scatter(year_list_2.index, year_list_2.values, color = "orange", marker = "^", label = "Total TV-Shows released in a year")
axes_2[1].set_title("TV-Shows Released Vs Releasing-Year")
axes_2[1].set_xlabel("Releasing-Year ------->")
axes_2[1].set_ylabel("Total TV-Shows released --------->")

axes_2[1].legend(loc = "upper left", fontsize = 9)

axes_2[1].grid(color = "gray", linestyle = ":", linewidth = 0.50)

plt.savefig("I:/ML_FOLDER/MATPLOTLIB_FOLDER/MatplotLib_Project/Netflix_Plot_4.jpg", dpi = 350, bbox_inches = "tight")

fig_3, axes_3 = plt.subplots(1, figsize = (10, 7))

width = 0.25

axes_3.plot(year_list.index, year_list.values, color = "orange",  linewidth = 2, linestyle = "--", marker = "o", label = "Total Movies released in a year")
axes_3.plot(year_list_2.index, year_list_2.values, color = "blue", linewidth = 1, linestyle = "--", marker = "^", label = "Total TV-Shows released in a year")

axes_3.set_title("Movies Released Vs TV-Shows Released in a year")
axes_3.set_xlabel("Year -------->")
axes_3.set_ylabel("Total Movies/TV-Shows Released -------->")

axes_3.legend(loc = "upper left", fontsize = 14)

axes_3.grid(color = "gray", linestyle = ":", linewidth = 0.50)

plt.tight_layout()
plt.savefig("I:/ML_FOLDER/MATPLOTLIB_FOLDER/MatplotLib_Project/Netflix_Plot_5.jpg", dpi = 350, bbox_inches = "tight")
plt.show()
