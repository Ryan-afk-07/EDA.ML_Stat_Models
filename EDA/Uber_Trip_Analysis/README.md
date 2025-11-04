<h1 align="center">Exploratory Data Analysis of Uber Trips in US</h1>
<div>
  <h2>Project/EDA Objective</h2>
  <p>
    To find out patterns and trends of Uber uptakes in US are present between months, days and hours. To also find out if there is a pattern in which certain Uber app groups are widely used in a specific location as well.
  </p>
</div>
<div>
  <h2>General Information, Dataset Metadata</h2>
  <p><strong>Primarily used with:</strong> <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" height=20 width=20>
    <img src="https://jupyter.org/assets/homepage/main-logo.svg" height=20 width=20></p>
  <strong>Dataset retrieved from <a href="https://www.kaggle.com/datasets/dkhalidashik/uber-trips-data">Kaggle</a></strong>
  <p></p>
  <table>
    <thead>
      <tr>
        <th>Column Name</th>
        <th>Description of Column</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Date/Time</td>
        <td>Date and Time of the booked Uber Trip</td>
        <td>Datetime</td>
        <td>Datetime Float</td>
      </tr>
      <tr>
        <td>Lat</td>
        <td>Latitude</td>
        <td>Geographical</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Lon</td>
        <td>Longitude</td>
        <td>Geographical</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Base</td>
        <td>Interpreted to be either the particular cab that Uber provides or the region in the city or country in which the pickup occurs. Either which groups can be interpreted for analysis</td>
        <td>String (combination of letters and numbers)</td>
        <td>String</td>
      </tr>
    </tbody>
  </table>
</div>
<div>
  <h2>Cleaning, Preprocessing and Transformation of Data</h2>
  <p>
    <strong>1. </strong> Data is presented with no nulls. Did not perform any null cleaning <br>
    <strong>2. </strong> Split datetime into columns for month, date/day and time for more precise trend analysis/exploration <br>
    <strong>3. </strong> Did a mapping of key_value for Month - primarily for visual purposes
    <strong>4. </strong> Data is huge. Hence merge results in 2 main datasets. 1 from Apr to Sep, 1 from Jan to Jun
  </p>
</div>
<div>
  <h2>Exploratory Data Analysis of Trends with Uber Trips</h2>
  <p>
    <strong>1. </strong> Created visualizations to view total number of trips in the region across months and days.
    <strong>2. </strong> Created visualizations to view number of trips in the region across months and days grouped by bases (get a good view of which base is heavily used)
  </p>
  <strong>Visualizations of trends</strong>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Uber_Trip_Analysis/Visualization1_tripcount_aprtosep.png">
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Uber_Trip_Analysis/Visualization2_tripcount_aprtosep.png">
</div>
<div>
  <h2>Observations, Insights</h2>
  <p>
    <strong>1. </strong> Trips are generally more at the later months (Sep) <br>
    <strong>2. </strong> Trips are generally more frequent at the later hours of the day <br>
    <strong>3. </strong> No strong correlation with regards to dates and Uber trips - Cabs/User is equally frequent across all days of the month <br>
    <strong>4. </strong> Bases B02512 and B02764 have vastly lesser Uber Pickup and trips across the months than the rest.
  </p>
</div>
<div>
  <h2>Exploratory Data Analysis of Location Trends for Uber Trips</h2>
  <strong>Visualizations</strong>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Uber_Trip_Analysis/Visualization3_location_aprtosep.png">
  <p>
    <strong>Observations, Findings</strong> <br>
    <strong>1. </strong> B02764 has a substantial amount of trips undertaken on September. It is also seen to be consistent at the middle region of the US city taken.
    <strong>2. </strong> 02682 is consistently plenty and picked up or used at similar locations from April to August. Only outlier would be September.
    <strong>3. </strong> B02617 is consistently picked up at rather consistent locations as well. However rates are rather varied (drops on Jun and Aug)
  </p>
</div>
