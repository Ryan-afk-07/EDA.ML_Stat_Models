<h1 align='center'>HR Predictive Analytics</h1>
<div>
  <h2>General Information</h2>
  <strong>Primarily analyzed with 
    <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" height=20 width=20>
    <img src="https://jupyter.org/assets/homepage/main-logo.svg" height=20 width=20>
  </strong>
  <strong>
    Dataset retrieved from <a href="https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?resource=download">Kaggle</a>
  </strong>
</div>

<div>
  <h2>Metadata, Dataset details</h2>
  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Satisfication Level</td>
        <td>Results of an individual's satisfication with the company/role</td>
        <td>Continuous</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Last Evaluation</td>
        <td>Self interpretation: Last Evaluation rating of individual's performance</td>
        <td>Continous</td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Number Project</td>
        <td>Self interpretation: Number of projects individual has assisted/contributed during his/her tenure with company</td>
        <td>Continuous</td>
        <td>Whole Number</td>
      </tr>
      <tr>
        <td>Average Monthly Hours</td>
        <td>Number of hours individual worked in a month (averaged out across entire period individual is with company)</td>
        <td>Continuous (mean></td>
        <td>Float</td>
      </tr>
      <tr>
        <td>Time Spent with Company</td>
        <td>Number of years individual spent with company (numbers likely truncated unless individual has spent significant time with company for the particular year - in which it will likely be rounded)</td>
        <td>Continuous</td>
        <td>Whole Number</td>
      </tr>
      <tr>
        <td>Work Accident</td>
        <td>If individual has sustained any work accidents during his/her tenure with the company</td>
        <td>Categorical (Binary)</td>
        <td>Binary/Boolean</td>
      </tr>
      <tr>
        <td>Left</td>
        <td>Individual has or has not left the company</td>
        <td>Categorical (Binary)</td>
        <td>Binary</td>
      </tr>
      <tr>
        <td>Promotion Last 5 Years</td>
        <td>Individual has or has not received a promotion in the last 5 years (or years in which he or she is employed for if less than 5)</td>
        <td>Categorical (Binary)</td>
        <td>Binary</td>
      </tr>
      <tr>
        <td>Department</td>
        <td>Department individual is in within the company</td>
        <td>Categorical</td>
        <td>String</td>
      </tr>
      <tr>
        <td>Salary</td>
        <td>Salary tier in which the individual is paid prior to his/her departure</td>
        <td>Categorical</td>
        <td>String</td>
      </tr>
    </tbody>
  </table>
</div>


<div>
  <h2>Cleaning, Transforming Steps and Processes</h2>
  <p><strong>Cleaning</strong>: Data was presented rather cleanly. No NAs or NULL values to be removed</p>
  <p><strong>Transformation</strong>: Changed/Updated the department and salary categorical strings into numerical format for ease in the feature selection/modeling stages</p>
</div>

<div>
  <h2>Exploratory Data Analysis</h2>
  <h3>Summarized Findings:</h3>
  <p>1. Satisfication Levels provide significant importance/influence to an individual's stay with the company</p>
  <p>2. Averge monthly hours for individuals whom left also show a bigger spread than those who did not leave</p>
  <p>3. For an individual, visualization shows that the likelihood of him/her leaving trends very critically to no once the 7 year mark is reach. It also shows an inclination of resignation towards the 5 year mark and thereafter an declination.</p>
  <p>4. There is not much insight that could be derived from whether evaluation scores or if the individual has been promoted that will result in him/her leaving the company</p>
  <strong>Snippet of Data Visualizations which explains the above findings</strong> <br>
  <img src="" height=100>
</div>

<div>
  <h2>Feature Selection</h2>
  <p><strong>Techniques used:</strong> Feature importance Profilling, Heatmap Correlation</p>
  <p><strong>Eventual features used</strong>: Satisfaction Level, Average Monthly Hours, Salary, Last 5 years promotion, Time Spend in Company</p> <br>
  <img src="" height=100>
</div>

<div>
  <h2>Classification Modelling to Predict Individual's departure</h2>
  <table>
    <thead>
      <tr>
        <th>Classification Model</th>
        <th>Additional Tuning</th>
        <th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Decision Tree Classifier</td>
        <td>No</td>
        <td>0.9726 (97.26%)</td>
      </tr>
      <tr>
        <td>Decision Tree Classifier</td>
        <td>Yes- Hyperparameter tuning (class weight, criterion, max_depth, min_samples_leaf, min_samples_split)</td>
        <td>0.9675 (96.75%)</td>
      </tr>
      <tr>
        <td>Random Forest Classifier</td>
        <td>No</td>
        <td>0.9795 (97.95%)</td>
      </tr>
      <tr>
        <td>Random Forest Classifier</td>
        <td>Yes - Hyperparameter tuning (n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, class_weight, n_jobs)</td>
        <td>0.9742 (97.42%)</td>
      </tr>
      <tr>
        <td>XGBoost Classifier</td>
        <td>No</td>
        <td>0.978 (97.8%)</td>
      </tr>
      <tr>
        <td>XGBoost Classifier</td>
        <td>Yes - Hyperparameter tuning (max_depth , min_child_weight, gamma, subsample, colsample_bytree, reg_alpha, learning_rate, n_estimators)</td>
        <td>0.9879 (98.79%)</td>
      </tr>
    </tbody>
  </table>
</div>
