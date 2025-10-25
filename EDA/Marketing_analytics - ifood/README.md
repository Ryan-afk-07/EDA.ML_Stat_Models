<h1 align='center'>Marketing Analytics of Ifood Company</h1>
<div>
  <h2>General Information, Dataset Metadata</h2>
  <p>Dataset retrieved from <a href="">Kaggle</a></p>
  <p><strong>Dataset 1: </strong>Personal Information of Customers, Marketing Campaign reviews</p>
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
        <td>ID</td>
        <td>Identification Number of customer - kept anonymous for privacy purposes I believe</td>
        <td>Categorical</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Year_Birth (Birth Year)</td>
        <td>Year of Birth for the individual</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Education</td>
        <td>Highest Level of Education for the particular individual</td>
        <td>Categorical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Marital_Status</td>
        <td>Relationship Status of the individual at the instance of visiting company</td>
        <td>Categorical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Income</td>
        <td>Yearly income of individual at the instance of visiting company</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Kidhome, Teenhome</td>
        <td>Number of Kids, Teens in household of person</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Dt_Customer</td>
        <td>Date in which the individual first became customer of company</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>Recency</td>
        <td>Days before in which individual last visited company</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases</td>
        <td>Number of Purchases individual has made from deals, webs, catalogs and stroes</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5</td>
        <td>Whether individual has accepted the marketing campaign (i.e. if the campaign has driven the individual to purchase the company's products)</td>
        <td>Binary</td>
        <td>Integer/Binary(0,1)</td>
      </tr>
      <tr>
        <td>Country</td>
        <td>Country individual resides in</td>
        <td>Geographical</td>
        <td>String</td>
      </tr>
    </tbody>
  </table>
  <p><strong>Dataset 2: </strong>Amount of Goods per category bought by customers</p>
  <table>
    <thead>
      <tr>Column/Columns Names</tr>
      <tr>Column/Columns Descriptions</tr>
      <tr>Data Type(s)</tr>
      <tr>Data Format(s)</tr>
    </thead>
    <tbody>
      <tr>
        <td>MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts. MntGoldProducts</td>
        <td>Amount of Wine Products, Fruit Products, Meat Products, Fish Products, Sweets and Gold purchased by the individual during the visit</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
      <tr>
        <td>MntTotal</td>
        <td>Total Amount of Products/Items purchased by individual</td>
        <td>Continuous</td>
        <td>Integer/Whole Number</td>
      </tr>
    </tbody>
  </table>
</div>
<div>
  <h2>Data Cleaning, Pre-processing, Transformation</h2>
  <p>
    <strong>1. </strong> Observed Traits of dataset using .info(), .describe(), .shape(), .nunique() <br>
    <strong>2. </strong> Removed all irrelevant na values <br>
    <strong>3. </strong> Performed transformation of following columns: <br>
    <pre>I. Updated Income column to float data format for ease of calculations in the later stages</pre> <br>
    <pre>II. Created a Dependents column from the sum of Teens and Kids.</pre> <br>
    <pre>III. Created a total Purchase and Total Amount column from summing all Purchase and Mnt named columns respectively </pre><br>
    <pre>IV. Created a totalCampaignAcc column by summing all the Marketing Campaigns that had successfully influenced individual to purchase</pre>
    <pre>V. Did one hot/label encoding for education and Marital Status for ease of visualization and modeling</pre>
  <strong>4. </strong> Dropped all irrelevant columns/non numerical columns for the purposes of subsequent visualizations.
  </p>
</div>
<div>
  <h2>Exploratory Data Analysis</h2>
  <p>Mainly did a correlation heatmap for the purposes of subsequent modeling of trends and patterns. Seemed to be able to visualize correlation of all columns in the dataset and if they have correlation with other parameters/columns</p>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Marketing_analytics%20-%20ifood/Heatmap_elements.png"> <br>
  <p>Background understanding from news/discussions, having a higher income will likely influence individuals to purchase more goods. Hence doing more visualizations to compare income with number of goods purchased. In addition, also viewing if having dependents affect ones willingness to purchase goods</p>
  <img src="https://github.com/Ryan-afk-07/EDA.ML_Stat_Models/blob/main/EDA/Marketing_analytics%20-%20ifood/Income_Goodspurchased_dependents.png">
</div>
<div>
  <h2>Observations </h2>
  <p>
    <strong>1. </strong> According to the heatmap visualization, Campaign effectiveness is rather strongly correlated with income, Amounts of Meat and Wine products purchased. <br>
    <strong>2. </strong> According to the heatmap visualization as well, Campaign effectiveness seems to be negatively related with Dependents. <br>
    <strong>3. </strong> Based on the scatter plot, income does seem to correlate with number of goods purchased (Total). <br>
    <strong>4. </strong> Dependents does really relate to the number of goods purchased from ifood. Customers with more dependents do seem to have a lower range of goods purchased.
  </p>
</div>
