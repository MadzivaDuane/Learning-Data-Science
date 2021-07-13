#NHL
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities=pd.read_html("assets/wikipedia_data.html")[1]
cities_df=cities.iloc[:-1,[0,3,5,6,7,8]]
cities_df = cities_df.rename(columns = {"Population (2016 est.)[8]": "Population"})

#clean columns of numeric and character values 
cities_df['NFL'] = cities_df['NFL'].str.replace(r"\[.*\]", "")
cities_df['MLB'] = cities_df['MLB'].str.replace(r"\[.*\]", "")
cities_df['NBA'] = cities_df['NBA'].str.replace(r"\[.*\]", "")
cities_df['NHL'] = cities_df['NHL'].str.replace(r"\[.*\]", "")
league = 'NHL'
def nhl_correlation(): 
    team_df = cities_df[league].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team_df['Metropolitan area'] = cities_df['Metropolitan area']
    team_df = pd.melt(team_df, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("",np.nan).replace("—",np.nan).dropna()
    team_df = team_df.rename(columns = {"value":"team"})
    team_df = pd.merge(team_df,cities_df,how='left',on = 'Metropolitan area')
    team_df = team_df[['Metropolitan area', 'team', 'Population']]
    team_df = team_df.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team_df['team']= team_df['team'].str.replace('[\w.]*\ ','')
    
    #clean data
    #remove unnecessary rows
    nhl_df=pd.read_csv("assets/nhl.csv")
    nhl_df = nhl_df[~nhl_df['team'].str.contains('Division')]  #~ filters out
    nhl_df = nhl_df[nhl_df['year']==2018]
    nhl_df = nhl_df[['team', 'W', 'L']]
    nhl_df['team'] = nhl_df['team'].str.replace(r'\*',"")
    nhl_df[["W", "L"]] = nhl_df[["W", "L"]].apply(pd.to_numeric)
    nhl_df['W/L Ratio'] = nhl_df['W']/(nhl_df['W']+nhl_df['L'])
    nhl_df['team'] = nhl_df['team'].str.replace('[\w.]* ','')
    
    nhl_merged_df = pd.merge(team_df, nhl_df, how = 'outer', on = 'team')
    nhl_merged_df=nhl_merged_df.groupby('Metropolitan area').agg({'W/L Ratio': np.nanmean, 'Population': np.nanmean})
    #raise NotImplementedError()
    
    population_by_region = nhl_merged_df['Population'] # pass in metropolitan area population from cities
    win_loss_by_region = nhl_merged_df['W/L Ratio'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]
    
    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#NBA
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities=pd.read_html("assets/wikipedia_data.html")[1]
cities_df=cities.iloc[:-1,[0,3,5,6,7,8]]
cities_df = cities_df.rename(columns = {"Population (2016 est.)[8]": "Population"})

#clean columns of numeric and character values 
cities_df['NFL'] = cities_df['NFL'].str.replace(r"\[.*\]", "")
cities_df['MLB'] = cities_df['MLB'].str.replace(r"\[.*\]", "")
cities_df['NBA'] = cities_df['NBA'].str.replace(r"\[.*\]", "")
cities_df['NHL'] = cities_df['NHL'].str.replace(r"\[.*\]", "")
league = 'NBA'

def nba_correlation(): 
    team_df = cities_df[league].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team_df['Metropolitan area'] = cities_df['Metropolitan area']
    team_df = pd.melt(team_df, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("",np.nan).replace("—",np.nan).dropna()
    team_df = team_df.rename(columns = {"value":"team"})
    team_df = pd.merge(team_df,cities_df,how='left',on = 'Metropolitan area')
    team_df = team_df[['Metropolitan area', 'team', 'Population']]
    team_df = team_df.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team_df['team']= team_df['team'].str.replace('[\w.]*\ ','')
    
    #clean data
    nba_df=pd.read_csv("assets/nba.csv")
    nba_df = nba_df[nba_df['year']==2018]
    nba_df = nba_df[['team', 'W/L%']]
    nba_df['team'] = nba_df['team'].str.replace(r'\*',"")
    nba_df['team'] = nba_df['team'].str.replace(r'\(\d*\)',"")
    nba_df['team'] = nba_df['team'].str.replace(r'[\xa0]',"")
    nba_df['W/L%'] = nba_df['W/L%'].apply(pd.to_numeric)
    nba_df['team'] = nba_df['team'].str.replace('[\w.]* ','')

    nba_merged_df = pd.merge(team_df, nba_df, how = 'outer', on = 'team')
    nba_merged_df = nba_merged_df.groupby('Metropolitan area').agg({'W/L%': np.nanmean, 'Population': np.nanmean})

    return stats.pearsonr(nba_merged_df['Population'], nba_merged_df['W/L%'])[0]
    
#MLB
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities=pd.read_html("assets/wikipedia_data.html")[1]
cities_df=cities.iloc[:-1,[0,3,5,6,7,8]]
cities_df = cities_df.rename(columns = {"Population (2016 est.)[8]": "Population"})

#clean columns of numeric and character values 
cities_df['NFL'] = cities_df['NFL'].str.replace(r"\[.*\]", "")
cities_df['MLB'] = cities_df['MLB'].str.replace(r"\[.*\]", "")
cities_df['NBA'] = cities_df['NBA'].str.replace(r"\[.*\]", "")
cities_df['NHL'] = cities_df['NHL'].str.replace(r"\[.*\]", "")
league = 'MLB'
def mlb_correlation(): 
    team_df = cities_df[league].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team_df['Metropolitan area'] = cities_df['Metropolitan area']
    team_df = pd.melt(team_df, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("",np.nan).replace("—",np.nan).dropna()
    team_df = team_df.rename(columns = {"value":"team"})
    team_df = pd.merge(team_df,cities_df,how='left',on = 'Metropolitan area')
    team_df = team_df[['Metropolitan area', 'team', 'Population']]
    team_df = team_df.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team_df['team']=team_df['team'].str.replace('\ Sox','Sox')
    team_df['team']= team_df['team'].str.replace('[\w.]*\ ','')
    
    mlb_df=pd.read_csv("assets/mlb.csv")
    mlb_df = mlb_df[mlb_df['year']==2018]
    mlb_df = mlb_df[['team', 'W-L%']]
    mlb_df['team'] = mlb_df['team'].str.replace(r'\*',"")
    mlb_df['team'] = mlb_df['team'].str.replace(r'\(\d*\)',"")
    mlb_df['team'] = mlb_df['team'].str.replace(r'[\xa0]',"")
    mlb_df['W-L%'] = mlb_df['W-L%'].apply(pd.to_numeric)
    mlb_df['team']= mlb_df['team'].str.replace('\ Sox','Sox')
    mlb_df['team'] = mlb_df['team'].str.replace('[\w.]* ','')

    mlb_merged_df = pd.merge(team_df, mlb_df, how = 'outer', on = 'team')
    mlb_merged_df=mlb_merged_df.groupby('Metropolitan area').agg({'W-L%': np.nanmean, 'Population': np.nanmean})
    #raise NotImplementedError()
    
    population_by_region = mlb_merged_df['Population']# pass in metropolitan area population from cities
    win_loss_by_region = mlb_merged_df['W-L%'] # pass in win/loss ratio from mlb_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#NFL
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities=pd.read_html("assets/wikipedia_data.html")[1]
cities_df=cities.iloc[:-1,[0,3,5,6,7,8]]
cities_df = cities_df.rename(columns = {"Population (2016 est.)[8]": "Population"})

#clean columns of numeric and character values 
cities_df['NFL'] = cities_df['NFL'].str.replace(r"\[.*\]", "")
cities_df['MLB'] = cities_df['MLB'].str.replace(r"\[.*\]", "")
cities_df['NBA'] = cities_df['NBA'].str.replace(r"\[.*\]", "")
cities_df['NHL'] = cities_df['NHL'].str.replace(r"\[.*\]", "")
league = 'NFL'

def nfl_correlation(): 
    team_df = cities_df[league].str.extract('([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)([A-Z]{0,2}[a-z0-9]*\ [A-Z]{0,2}[a-z0-9]*|[A-Z]{0,2}[a-z0-9]*)')
    team_df['Metropolitan area'] = cities_df['Metropolitan area']
    team_df = pd.melt(team_df, id_vars=['Metropolitan area']).drop(columns=['variable']).replace("",np.nan).replace("—",np.nan).dropna()
    team_df = team_df.rename(columns = {"value":"team"})
    team_df = pd.merge(team_df,cities_df,how='left',on = 'Metropolitan area')
    team_df = team_df[['Metropolitan area', 'team', 'Population']]
    team_df = team_df.astype({'Metropolitan area': str, 'team': str, 'Population': int})
    team_df['team']= team_df['team'].str.replace('[\w.]*\ ','')
    
    nfl_df=pd.read_csv("assets/nfl.csv")
    nfl_df = nfl_df[nfl_df['year']==2018]
    nfl_df = nfl_df[~nfl_df['team'].str.contains('AFC')]  #~ filters out
    nfl_df = nfl_df[~nfl_df['team'].str.contains('NFC')]  #~ filters out
    nfl_df = nfl_df[['team', 'W-L%']]
    nfl_df['team'] = nfl_df['team'].str.replace(r'\*',"")
    nfl_df['team'] = nfl_df['team'].str.replace(r'\+',"")
    nfl_df['team'] = nfl_df['team'].str.replace(r'\(\d*\)',"")
    nfl_df['team'] = nfl_df['team'].str.replace(r'[\xa0]',"")
    nfl_df['W-L%'] = nfl_df['W-L%'].apply(pd.to_numeric)
    nfl_df['team'] = nfl_df['team'].str.replace('[\w.]* ','')

    nfl_merged_df = pd.merge(team_df, nfl_df, how = 'outer', on = 'team')
    nfl_merged_df=nfl_merged_df.groupby('Metropolitan area').agg({'W-L%': np.nanmean, 'Population': np.nanmean})
    #raise NotImplementedError()
    
    population_by_region = nfl_merged_df['Population'] # pass in metropolitan area population from cities
    win_loss_by_region = nfl_merged_df['W-L%'] # pass in win/loss ratio from nfl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#Are two sport teams in different sports in the same area likely to perform the same 
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
nhl_df=pd.read_csv("assets/nhl.csv")
nba_df=pd.read_csv("assets/nba.csv")
nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def sports_team_performance():
    raise NotImplementedError()
    
    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k:np.nan for k in sports}, index=sports)
    
    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values
