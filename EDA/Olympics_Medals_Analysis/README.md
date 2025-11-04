<h1 align='center'>EDA of Olympic Medals by Country (Summer/Winter)</h1>
<div>
  <h2>Project/EDA Objective</h2>
  <p>To understand the trend and growth of the Summer and Winter olympic scene from 1990s to 2014. To find out patterns between economic factors (size, population, GDP) with performance of a country for olympic games. Finding out if there is a possibility for modeling as well.</p>
</div>
<div>
  <h2>General Information, Dataset Metadata</h2>
  <p><strong>Primarily used with:</strong> <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" height=20 width=20>
    <img src="https://jupyter.org/assets/homepage/main-logo.svg" height=20 width=20></p>
  <p><strong>Retrieved datasets from: </strong><a href="https://www.kaggle.com/datasets/the-guardian/olympic-games/data">Kaggle</a></p>
  <h3>Dictionary Dataset</h3>
  <table>
    <thead>
      <tr>
        <th>Column Name</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Country Name</td>
        <td>Full name of the country</td>
        <td>Geographical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Country Code</td>
        <td>3 Letter country code - for purposes of reference with other datasets containing 3 letter country codes</td>
        <td>Geographical (Short form)</td>
        <td>String/Char</td>
      </tr>
      <tr>
        <td>Population</td>
        <td>Number of citizens in country (updated 2014)</td>
        <td>Continuous</td>
        <td>Whole Number</td>
      </tr>
      <tr>
        <td>GDP Per Capita</td>
        <td>GDP for the country (updated latest 2014)</td>
        <td>Continuous</td>
        <td>Float</td>
      </tr>
    </tbody>
  </table>
  <h3>Summer and Winter Datasets</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Columns of note</th>
        <th>Link to dataset</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>Summer Olympic Games (1986 to 2014)</th>
        <th>Year, Sport, Discipline, Country, Gender, Medal</th>
        <th><a href="">Summer csv link</a></th>
      </tr>
      <tr>
        <th>Winter Olympic Games (1986 to 2014)</th>
        <th>Year, Sport, Discipline, Country, Gender, Medal</th>
        <th><a href="">Winter csv link</a></th>
      </tr>
    </tbody>
  </table>
</div>
<div>
  <h2>Data Cleaning, Transformation</h2>
  <strong>1. </strong><p>Cleaned up rows that contained NA values</p>
  <strong>2. </strong><p>Updated Year column (YYYY-MM-DD)for both summer and winter datasets into just Datetime (Year) format, for ease of transformation and subsequent visualization</p>
  <strong>3. </strong><p>Merged Ref Dictionary Dataset with the Summer Olympic and Winter Olympic count datasets</p>
</div>
<div>
  <h2>Exploratory Data Visualization</h2>
  <strong>1. Created chloropleth maps for summer and winter olympics to show spread of Olympic medal counts </strong> <br>
  <strong>Summer Olympic games</strong>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Olympics_Medals_Analysis/newplot.png">
  <p>You may view the entire cloropleth visualization gif via this <a href="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Olympics_Medals_Analysis/summer.html">Link</a></p>
  <strong>Winter Olympic games</strong>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Olympics_Medals_Analysis/newplot_winter.png">
  <p>You may view the entire chloropleth map visualization gif via this <a href="">Link</a></p>
  <p>Findings: <br>
    1. Prior to 1990s, Asia did not have a huge representation in the Summer and Winter Olympic Games. Was mostly America and Europe <br>
    2. Number of medals issued per each subsequent Summer olympic games increases. This would most likely be due to the inclusion of more varied sport disciplines
    3. America is a consistent top few performer across all Summer Olympic games
    4. There was evidently lesser medals given in the Winter games than Summer games. Most likely due to then (and now) winter games having rather lesser disciplines than summer games.
    5. Medal distribution was rather evenly distributed across. There seems to not be a trend in which one country consistently performed.
  </p>
  <strong>2. Created Correlation Heatmap in an effort to find out if a change in medal count may be related to a update in GDP or population</strong>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Olympics_Medals_Analysis/Correlation_medals.png">

  <p>
    Findings: <br>
    1. Heatmap/Correlation Map does show population and GDP Per Capita having some sort of positive correlation or relation with number of medals attained by the country for that year
  </p>
  
</div>
