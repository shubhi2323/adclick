document.addEventListener('DOMContentLoaded', () => {
  let csvData = []; // To store parsed CSV data

  // Handle file upload and parse CSV
  document.getElementById('upload-button').addEventListener('click', () => {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    const uploadStatus = document.getElementById('upload-status');

    if (file) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: function (results) {
          csvData = results.data;
          uploadStatus.innerText = `Uploaded: ${file.name} (Rows: ${csvData.length})`;
          uploadStatus.style.color = 'green';
          console.log('Parsed CSV Data:', csvData);
          updateCharts(csvData); // Update charts dynamically
        },
        error: function (error) {
          uploadStatus.innerText = `Error parsing file: ${error.message}`;
          uploadStatus.style.color = 'red';
        },
      });
    } else {
      uploadStatus.innerText = 'Please upload a valid CSV file.';
      uploadStatus.style.color = 'red';
    }
  });

  // Function to update charts based on CSV data
  function updateCharts(data) {
    const ageGroups = { '18-25': 0, '26-35': 0, '36-45': 0, '46-55': 0, '56+': 0 };
    const timeOfDayClicks = { Morning: 0, Afternoon: 0, Evening: 0, Night: 0 };

    // Process data from CSV
    data.forEach(row => {
      const age = parseInt(row.age, 10);
      const time = row.time_of_day;
      const click = parseInt(row.click, 10);

      // Increment age group counts
      if (age >= 18 && age <= 25) ageGroups['18-25']++;
      else if (age >= 26 && age <= 35) ageGroups['26-35']++;
      else if (age >= 36 && age <= 45) ageGroups['36-45']++;
      else if (age >= 46 && age <= 55) ageGroups['46-55']++;
      else if (age > 55) ageGroups['56+']++;

      // Increment time of day click counts
      if (click === 1 && timeOfDayClicks[time] !== undefined) {
        timeOfDayClicks[time]++;
      }
    });

    // Update charts dynamically
    renderCharts(Object.values(ageGroups), Object.values(timeOfDayClicks));
  }

  // Function to render charts
  function renderCharts(ageData, timeData) {
    // Age Distribution Chart
    const ctx1 = document.getElementById('chart1').getContext('2d');
    new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: ['18-25', '26-35', '36-45', '46-55', '56+'],
        datasets: [{
          label: 'Age Distribution',
          data: ageData,
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
        }]
      }
    });

    // Time of Day Click Rate Chart
    const ctx2 = document.getElementById('chart2').getContext('2d');
    new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: ['Morning', 'Afternoon', 'Evening', 'Night'],
        datasets: [{
          label: 'Click Rate by Time',
          data: timeData,
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
        }]
      }
    });
  }

  // Predict functionality 
  document.getElementById('prediction-form').addEventListener('submit', (e) => {
    e.preventDefault();

    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const device = document.getElementById('device').value;
    const time = document.getElementById('time').value;

    if (!age || !gender || !device || !time) {
      document.getElementById('prediction-result').innerText = 'Please fill out all fields!';
      document.getElementById('prediction-result').style.color = 'red';
      return;
    }

    const mockPrediction = Math.random() > 0.5 ? 'Yes, User will click!' : 'No, User will not click.';
    document.getElementById('prediction-result').innerText = `Prediction: ${mockPrediction}`;
    document.getElementById('prediction-result').style.color = 'blue';
  });
});