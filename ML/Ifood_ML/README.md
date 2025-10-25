<h1 align="center">Marketing Analytics of Ifood Company (Modeling)</h1>
<div>
  <h2>General Information, Dataset Metadata</h2>
  <p>
    Dataset is the same as that used for the EDA in the EDA folder <br>
    You may head to <a href="">this folder</a> for the table that lists all relevant information.
  </p>
</div>
<div>
  <h2>Models</h2>
  <table>
    <thead>
      <tr>
        <th>Analysis</th>
        <th>Models used</th>
        <th>Accuracy</th>
        <th>Takeaways</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Presence of relationships between TotalPurchases/Total Amount and Demographical Information (Best parameters)</td>
        <td>Linear regression</td>
        <td>R2 score: 0.2249, MSE: 448577.9952</td>
        <td>Demographical Parameters were specifically considered and chosen. Best/Most related parameters that could show correlated with purchases. <br>
          <strong>Chosen parameters: </strong> Income, Recency, Complains, Number of Web Visits per month. <br>
        However, model is rather weak still. <strong>Conclusion: </strong> Dataset does now show adequately in general a relation to Number of good purchases/amounts</td>
      </tr>
      <tr>
        <td>Prediction of campaign acceptance from setup parameters (5 models for 5 different campaigns)</td>
        <td>Logistic Regression</td>
        <td>
          Campaign 1: Accuracy - 0.9394 (93.94%) <br>
          Campaign 2: Accuracy - 0.9890 (98.9%) <br>
          Campaign 3: Accuracy - 0.9226 (92.26&) <br>
          Campaign 4: Accuracy - 0.9336 (93.36%) <br>
          Campaign 5: Accuracy - 0.9413 (94.13%) <br>
        </td>
        <td>Accuracy of logistic regression may be good due to an imbalanced data. Performing balancing of data through oversampling or undersampling seems to reduce its performance.</td>
      </tr>
    </tbody>
  </table>
</div>
