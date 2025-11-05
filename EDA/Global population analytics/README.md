<h1 align='center'>World Demographics, Population Analysis/EDA</h1>
<div>
  <h2>Project/EDA Objectives</h2>
  <p>
    Project/EDA seeks to find out and visualize relationships between specific demographic factors and economic/environmental factors (i.e. if an increase in population has a relation to higher environmental gases produced). Exploration is also performed with the intention to see if future modelling (regression, trees) is possible and could be explored as well.
  </p>
</div>
<div>
  <h2>General Information, Dataset Metadata</h2>
  <p><strong>Primarily used with:</strong> <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" height=20 width=20>
    <img src="https://jupyter.org/assets/homepage/main-logo.svg" height=20 width=20></p>
  <strong>Dataset retrieved from <a href="">Kaggle</a></strong>
  <p></p>
  <table>
    <thead>
      <tr>
        <th>Column Name</th>
        <th>Column Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Country</td>
        <td>Name of Country</td>
        <td>Geographical/Categorical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Density</td>
        <td>Population Density (denoted in P/Km2)</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Abbreviation</td>
        <td>Country Abbreviation</td>
        <td>Geographical/Categorical</td>
        <td>Character(2 char)</td>
      </tr>
      <tr>
        <td>Agricultural Land</td>
        <td>Percentage of total land in country available/used for agricultural purposes</td>
        <td>Continuous</td>
        <td>Float/Decimal (Percentage)</td>
      </tr>
      <tr>
        <td>Land Area</td>
        <td>Total Land present in country</td>
        <td>Continuous</td>
        <td>String (Raw) - meant to be Integer</td>
      </tr>
      <tr>
        <td>Armed Forces Size</td>
        <td>Size/Amount of Soldiers enlisted/serving in the country's armed forces</td>
        <td>Continuous</td>
        <td>String (Raw Data) - meant to be Integer</td>
      </tr>
      <tr>
        <td>Birth Rate</td>
        <td>Birth rate of country</td>
        <td>Continuous</td>
        <td>Float (percentage)</td>
      </tr>
      <tr>
        <td>Calling Code</td>
        <td>Codes of country - particularly for phone calls across countries/zones</td>
        <td>Geographical/Categorical</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Capital/Major City</td>
        <td>Capital of the country</td>
        <td>Geographical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>CO2 Emissions</td>
        <td>Emissions of CO2 produced by the country</td>
        <td>Continuous (Geography affected)</td>
        <td>Whole Number/Integer</td>
      </tr>
      <tr>
        <td>Currency-Code</td>
        <td>Code for currency of the country</td>
        <td>Geographical/Categorical</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Gross primary and tertiary education enrollment </td>
        <td>Gross perentage of citizens in country with primary education and tertiary education</td>
        <td>Geographical/Categorical</td>
        <td>Float (percentage)</td>
      </tr>
      <tr>
        <td>Life Expectancy</td>
        <td>Average age a citizen in a country lives until death</td>
        <td>Continuous (Geographical dependent)</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>GDP</td>
        <td>Gross Domestic Product of a country</td>
        <td>Continuous</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Minimum Wage</td>
        <td>Designated lowest salary paid to a worker in the specified country</td>
        <td>Continuous</td>
        <td>Float (possess a $ symbol at the start)</td>
      </tr>
      <tr>
        <td>Official Language</td>
        <td>Main/Most commonly used language in the country</td>
        <td>Categorical/Geographical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Population</td>
        <td>Number of residents in the country</td>
        <td>Continuous (grouped by geography)</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Population: Labor force participation</td>
        <td>Percentage of residents in the country that are actively working or part of the labor force</td>
        <td>Continuous</td>
        <td>Float (percentage)</td>
      </tr>
      <tr>
        <td>Unemployment Rate</td>
        <td>Percentage of labor market/eligible residents actively seeking a job/not working</td>
        <td>Continuous</td>
        <td>Float (percentage)</td>
      </tr>
      <tr>
        <td>Urban Population</td>
        <td>Population of residents living in rather urbanized areas</td>
        <td>Continuous</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Latitude, Longitude</td>
        <td>Position of country in the world (length, height)</td>
        <td>Geographical</td>
        <td>Integer/Float</td>
      </tr>
    </tbody>
  </table>
</div>
<div>
  <h2>Pre-processing, Cleaning, Transformation of Data</h2>
  <p>
    <strong>1. </strong> Data has no null values when retrieved raw. No cleaning is actually done <br>
    <strong>2. </strong> Data types for certain integer intended columns are given as strings, with unnecessary symbols added either before or after the important values. Did data format cleaning to ensure columns that are meant to be strings are strings, those that are integers, and those that are meant to be floats <br>
    <strong>3. </strong> Checked the information of each column (max, min, count, mean etc) using .info and .describe
  </p>
</div>
<div>
  <h2>Exploratory Data Analysis, Visualization</h2>
  <p>
    <strong>1. </strong> Performed various barplot visualizations to view the top few countries with big populations, as well as top economic and environmental factors - each visualization is coupled with their population/land size to view if performing factors are affected by size or number of people in the country. <br>
    <strong>2. </strong> Did a generic heatmap/correlation heatmap to check if certain parameters (social, economic, environmental) are correlated or are in relation with another factor
  </p>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Global%20population%20analytics/population_visualization.png">
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Global%20population%20analytics/top_10_econ_environ.png">
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Global%20population%20analytics/correlationmap_demograpics.png">
</div>
<div>
  <h2>EDA Observations</h2>
  <p>
    <strong>1. </strong> In terms of population metrics (armed force size, labor force and urban force) - China and India are consistently up in the top 10. US as well. But not throughout all tiers of population <br>
    <strong>2. </strong> Top countries with positive economic and environmental do not all have big populations. <br>
    <strong>3. </strong> Countries with high population density (i.e. Monaco and Singapore), do not necessarily have major land area. <br>
    <strong>4. </strong> According to the correlation map, population related metrics are strongly positively correlated with each other, as well as certain demographical factors like GDP and CO2 emissions. <br>
    <strong>5. </strong> Mortality and Fertility rate is negatively correlated with educational factors, as well as life expectancy and physicians count
  </p>
</div>
