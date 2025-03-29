// Main JavaScript for AI Co-Scientist

document.addEventListener('DOMContentLoaded', function() {
    // Enable all tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Set the current year in footer
    const currentYearSpan = document.getElementById('current-year');
    if (currentYearSpan) {
        currentYearSpan.textContent = new Date().getFullYear();
    }
    
    // Form validation enhancement
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
});

// Animate elements when they come into view
function animateOnScroll() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        
        if (elementPosition < windowHeight - 50) {
            element.classList.add('animate__animated', 'animate__fadeInUp');
            element.style.opacity = 1;
        }
    });
}

// Add animation classes to elements
function setupAnimations() {
    const elementsToAnimate = document.querySelectorAll('.card, h1, h2, h3, h4, .lead');
    
    elementsToAnimate.forEach(element => {
        element.classList.add('animate-on-scroll');
        element.style.opacity = 0;
    });
    
    // Run once on load
    animateOnScroll();
    
    // Add scroll listener
    window.addEventListener('scroll', animateOnScroll);
}

// Initialize animations if not on mobile
if (window.innerWidth > 768) {
    window.addEventListener('load', setupAnimations);
}
