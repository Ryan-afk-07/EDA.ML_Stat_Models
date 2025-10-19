<h1 align="center">Exploratory Data Analysis of Uber Trips in US</h1>

<div>
  <h2>General Information, Dataset Metadata</h2>
  <strong>Dataset retrieved from <a href="https://www.kaggle.com/datasets/dkhalidashik/uber-trips-data">Kaggle</a></strong>
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
</div>
