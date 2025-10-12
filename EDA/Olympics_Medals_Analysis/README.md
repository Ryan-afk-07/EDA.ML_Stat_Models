<h1 align='center'>EDA of Olympic Medals by Country (Summer/Winter)</h1>

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
</div>
