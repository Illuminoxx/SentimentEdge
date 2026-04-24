// =========================
// KPI & Distribution Updates
// =========================
// Update KPI bars

function loadKPIs() {
  fetch("http://127.0.0.1:5000/api/kpis")
    .then(res => res.json())
    .then(data => {
      document.getElementById('kpi-acc').innerText = data.accuracy + "%";
      document.getElementById('kpi-f1').innerText = data.f1;
      document.getElementById('kpi-finbert').innerText = data.finbert_acc + "%";

      document.getElementById('dist-pos').style.height = data.positive + "%";
      document.getElementById('dist-neu').style.height = data.neutral + "%";
      document.getElementById('dist-neg').style.height = data.negative + "%";
    });
}

// Update pipeline stages
function loadPipeline() {
  fetch("http://127.0.0.1:5000/api/status")
    .then(res => res.json())
    .then(data => {
      data.stages.forEach(stage => {
        document.getElementById(`${stage.id}-status`).innerText = stage.status;
        document.getElementById(`${stage.id}-bar`).style.width = stage.progress + "%";
      });
      document.getElementById("pipeline-overall").innerText = data.overall + "%";
    });
}

// Refresh every 1-2 seconds
s// Refresh KPIs & live tweets every 2s
setInterval(() => {
  fetch("http://127.0.0.1:5000/api/kpis")
    .then(res => res.json())
    .then(data => {
      document.getElementById('kpi-acc').innerText = data.accuracy + "%";
      document.getElementById('kpi-f1').innerText = data.f1;
      document.getElementById('dist-pos').style.height = data.positive + "%";
      document.getElementById('dist-neu').style.height = data.neutral + "%";
      document.getElementById('dist-neg').style.height = data.negative + "%";
    });

  fetch("http://127.0.0.1:5000/api/live")
    .then(res => res.json())
    .then(data => {
      const feed = document.getElementById("live-feed");
      feed.innerHTML = "";
      data.slice(0,20).forEach(item => {
        const div = document.createElement("div");
        div.innerHTML = `<b>${item.finbert.label}</b>: ${item.text}`;
        feed.appendChild(div);
      });
    });
}, 2000);
// Check pipeline status every 2 seconds
setInterval(loadStatus, 2000);}

setInterval(loadStatus, 2000);
function updatePipelineDots() {
  fetch("http://127.0.0.1:5000/api/status")
    .then(res => res.json())
    .then(data => {
      const stages = data.stages;

      stages.forEach(stage => {
        const dots = document.querySelectorAll(".pipe-step .pipe-dot");
        dots.forEach(dot => {
          if (dot.innerText.trim() === stage.id) {
            // Remove old classes
            dot.classList.remove("done","active","idle");
            // Assign class based on status
            if (stage.status === "DONE") dot.classList.add("done");
            else if (stage.status === "RUNNING") dot.classList.add("active");
            else dot.classList.add("idle");
          }
        });
      });
    })
    .catch(err => console.error("PIPELINE DOTS ERROR:", err));
}

// Refresh dots every 1 second
setInterval(updatePipelineDots, 1000);