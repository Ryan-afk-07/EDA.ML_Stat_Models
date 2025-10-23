<h1 align="center">Exploratory Data Analysis of Netflix Films/Ratings</h1>

<div>
  <h2>General Information, Dataset Metadata</h2>
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
        <td>Title</td>
        <td>Title of Netflix Movie/Movie present in Netflix</td>
        <td>Descriptive</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Genre</td>
        <td>Genre of Netflix Movie/Movie present, listed in Netflix</td>
        <td>Categorical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Premiere</td>
        <td>Premiere date of the Netflix movie/Listed movie</td>
        <td>Date (Continuous)</td>
        <td>DateTime</td>
      </tr>
      <tr>
        <td>Runtime</td>
        <td>Length of Netflix movie/Listed movie</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>IMDB Score</td>
        <td>Score of Netflix movie/Listed movie given by IMDB</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Languages</td>
        <td>Main languages Neflix film/Listed film is filmed in/conveyed</td>
        <td>Categorical</td>
        <td>String</td>
      </tr>
    </tbody>
  </table>
</div>
<div>
  <h2>Data Preprocessing, Cleaning and Transformation</h2>
  <p>
    <strong>1. </strong> Data is encoded or recorded in a 'latin-1' format. Conversion from this format to the utc-8 acceptable format in jupyter is performed. <br>
    <strong>2. </strong> Data is retrieved clean from Kaggle. No cleaning of NA values is performed. <br>
    <strong>3. </strong> Certain movies have multiple genres included. Each genre in a multiple genre string is denoted with a '/' char. Using str.split of the '/' string and further save those split genres into new columns <br>
    <strong>4. </strong> Retrieval of top 10 IMDB scored films in popular categories
  </p>
</div>
