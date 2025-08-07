// This is added functionality to allow typing with commas in the input field for numerical values
function formatNumberWithCommas(num) {
  return Number(num).toLocaleString();
}

function stripCommas(str) {
  return str.replace(/,/g, "");
}

function bindFormattedNumberInput(inputId, min, max) {
  const input = document.getElementById(inputId);

  input.addEventListener("input", () => {
    // Only allow digits and commas
    const raw = stripCommas(input.value);
    if (raw !== "" && !/^\d+$/.test(raw)) {
      input.value = ""; // Optional: clear invalid input
    }
  });

  input.addEventListener("blur", () => {
    const raw = stripCommas(input.value);
    if (raw === "") return;

    const num = Number(raw);
    if (!isNaN(num) && num >= min && num <= max) {
      input.value = formatNumberWithCommas(num);
    } else {
      input.value = "";
    }
  });
}

bindFormattedNumberInput("aum_input", 10000, 1000000);
bindFormattedNumberInput("cryptoassets", 0, 1000000);
bindFormattedNumberInput("credit_input", 0, 1000000);
