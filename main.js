document.addEventListener('DOMContentLoaded', function() {
    console.log("Patient Assistance System Initialized");

    // Add a smooth fade-in to all images on load
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.classList.add('animate__animated', 'animate__fadeIn');
    });

    // Tooltip initialization for Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTrigger) {
      return new bootstrap.Tooltip(tooltipTrigger)
    })
});