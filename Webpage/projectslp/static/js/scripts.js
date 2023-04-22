const option = document.getElementById('option');
  const uploadInputs = document.getElementById('upload-inputs');
  const textInputs = document.getElementById('text-inputs');
  const pdfDocument = document.getElementById('pdf-document');
  const text = document.getElementById('text');
  const submitBtn = document.getElementById('submit-btn');

  option.addEventListener('change', () => {
    if (option.value === 'upload') {
      uploadInputs.classList.remove('hidden');
      textInputs.classList.add('hidden');
      pdfDocument.required = true;
      submitBtn.classList.remove('hidden');
    } else if (option.value === 'text') {
      textInputs.classList.remove('hidden');
      uploadInputs.classList.add('hidden');
      pdfDocument.required = false;
      submitBtn.classList.remove('hidden');
    } else {
      uploadInputs.classList.add('hidden');
      textInputs.classList.add('hidden');
      pdfDocument.required = false;
      submitBtn.classList.add('hidden');
    }
  });