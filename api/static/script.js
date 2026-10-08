const fileInput = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");
const detectBtn = document.getElementById("detectBtn");

const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");

const loading = document.getElementById("loading");
const result = document.getElementById("result");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const time = document.getElementById("time");

let selectedFile = null;


// Choose image
chooseBtn.addEventListener("click", () => {
    fileInput.click();
});


// Image selected
fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    selectedFile = file;

    const imageURL = URL.createObjectURL(file);

    preview.src = imageURL;
    preview.style.display = "block";

    placeholder.style.display = "none";

    detectBtn.disabled = false;

    result.style.display = "none";
});


// Detect defect
detectBtn.addEventListener("click", async () => {

    if (!selectedFile) {
        return;
    }

    const formData = new FormData();

    formData.append("file", selectedFile);


    loading.style.display = "block";
    result.style.display = "none";

    detectBtn.disabled = true;


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }


        prediction.textContent =
            data.predicted_class.toUpperCase();

        confidence.textContent =
            `${(data.confidence * 100).toFixed(2)}%`;

        time.textContent =
            `${data.inference_time_ms} ms`;


        result.style.display = "block";


    } catch (error) {

        alert(error.message);

    } finally {

        loading.style.display = "none";

        detectBtn.disabled = false;
    }
});