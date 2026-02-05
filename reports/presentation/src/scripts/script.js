<<<<<<< HEAD
// Configuration
let currentSlide = 1;
const totalSlides = 15;
let isFullscreen = false;

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser la première slide
    showSlide(currentSlide);
    
    // Événements de navigation
    document.getElementById('prev-btn').addEventListener('click', prevSlide);
    document.getElementById('next-btn').addEventListener('click', nextSlide);
    
    // Événements clavier
    document.addEventListener('keydown', handleKeydown);
    
    // Événements des contrôles
    document.getElementById('fullscreen-btn').addEventListener('click', toggleFullscreen);
    document.getElementById('print-btn').addEventListener('click', printPresentation);
    
    // Événements de swipe pour mobiles
    setupSwipeEvents();
    
});

// Navigation entre slides
function showSlide(n) {
    // Masquer toutes les slides
    document.querySelectorAll('.slide-container').forEach(slide => {
        slide.classList.remove('slide-active');
    });
    
    // Afficher la slide demandée
    const slideToShow = document.getElementById(`slide-${n}`);
    if (slideToShow) {
        slideToShow.classList.add('slide-active');
        currentSlide = n;
        
        // Mettre à jour la barre de progression
        updateProgressBar();
        
        // Mettre à jour les numéros de slide
        updateSlideNumbers();
    }
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        showSlide(currentSlide + 1);
    } else {
        // Revenir à la première slide
        showSlide(1);
    }
}

function prevSlide() {
    if (currentSlide > 1) {
        showSlide(currentSlide - 1);
    } else {
        // Aller à la dernière slide
        showSlide(totalSlides);
    }
}

// Gestion des touches du clavier
function handleKeydown(e) {
    switch(e.key) {
        case 'ArrowRight':
        case ' ':
        case 'PageDown':
            nextSlide();
            e.preventDefault();
            break;
            
        case 'ArrowLeft':
        case 'PageUp':
            prevSlide();
            e.preventDefault();
            break;
            
        case 'Home':
            showSlide(1);
            e.preventDefault();
            break;
            
        case 'End':
            showSlide(totalSlides);
            e.preventDefault();
            break;
            
        case 'f':
        case 'F':
            toggleFullscreen();
            e.preventDefault();
            break;
            
        case 'p':
        case 'P':
            printPresentation();
            e.preventDefault();
            break;
    }
}

// Mise à jour de la barre de progression
function updateProgressBar() {
    const progress = ((currentSlide - 1) / (totalSlides - 1)) * 100;
    document.getElementById('progress-bar').style.width = `${progress}%`;
}

// Mise à jour des numéros de slide
function updateSlideNumbers() {
    document.querySelectorAll('.slide-number').forEach((element, index) => {
        element.textContent = `${index + 1}/${totalSlides}`;
    });
}

// Gestion du plein écran
function toggleFullscreen() {
    const elem = document.documentElement;
    
    if (!isFullscreen) {
        if (elem.requestFullscreen) {
            elem.requestFullscreen();
        } else if (elem.webkitRequestFullscreen) { /* Safari */
            elem.webkitRequestFullscreen();
        } else if (elem.msRequestFullscreen) { /* IE11 */
            elem.msRequestFullscreen();
        }
        isFullscreen = true;
        document.getElementById('fullscreen-btn').innerHTML = '<i class="fas fa-compress"></i>';
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) { /* Safari */
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) { /* IE11 */
            document.msExitFullscreen();
        }
        isFullscreen = false;
        document.getElementById('fullscreen-btn').innerHTML = '<i class="fas fa-expand"></i>';
    }
}

// Détection de sortie du mode plein écran
document.addEventListener('fullscreenchange', handleFullscreenChange);
document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
document.addEventListener('msfullscreenchange', handleFullscreenChange);

function handleFullscreenChange() {
    isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
    document.getElementById('fullscreen-btn').innerHTML = isFullscreen ? 
        '<i class="fas fa-compress"></i>' : '<i class="fas fa-expand"></i>';
}

// Impression
function printPresentation() {
    window.print();
}

// Gestes tactiles (swipe)
function setupSwipeEvents() {
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    document.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
}

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0) {
            // Swipe gauche -> slide suivante
            nextSlide();
        } else {
            // Swipe droite -> slide précédente
            prevSlide();
        }
    }
}

// Navigation par clic sur la barre de progression
document.getElementById('progress-bar').addEventListener('click', function(e) {
    const progressBar = e.target;
    const clickPosition = e.offsetX;
    const progressBarWidth = progressBar.offsetWidth;
    const percentage = clickPosition / progressBarWidth;
    const slideNumber = Math.ceil(percentage * totalSlides);
    showSlide(slideNumber);
});

// Effets visuels supplémentaires
document.querySelectorAll('.objective-card, .data-card, .recommendation-card, .team-member').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.zIndex = '10';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.zIndex = '1';
    });
});

// Initialisation des images de secours
function setupFallbackImages() {
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('error', function() {
            const initials = this.alt.split(' ').map(n => n[0]).join('');
            this.src = `https://via.placeholder.com/150/667eea/ffffff?text=${initials}`;
        });
    });
}

// Exporter les fonctions pour le débogage
window.presentation = {
    nextSlide,
    prevSlide,
    showSlide,
    toggleFullscreen,
    printPresentation,
    currentSlide: () => currentSlide,
    totalSlides: () => totalSlides
};

console.log('Présentation CO₂ Seattle chargée avec succès!');
console.log('Commandes disponibles:');
console.log(' - Flèches gauche/droite: Navigation');
console.log(' - Espace/PageDown: Slide suivante');
console.log(' - PageUp: Slide précédente');
console.log(' - F: Plein écran');
console.log(' - P: Imprimer');
=======
// Configuration
let currentSlide = 1;
const totalSlides = 15;
let isFullscreen = false;
let autoPlayInterval = null;
const autoPlayDelay = 5000; // 10 secondes

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser la première slide
    showSlide(currentSlide);
    
    // Événements de navigation
    document.getElementById('prev-btn').addEventListener('click', prevSlide);
    document.getElementById('next-btn').addEventListener('click', nextSlide);
    
    // Événements clavier
    document.addEventListener('keydown', handleKeydown);
    
    // Événements des contrôles
    document.getElementById('fullscreen-btn').addEventListener('click', toggleFullscreen);
    document.getElementById('print-btn').addEventListener('click', printPresentation);
    
    // Événements de swipe pour mobiles
    setupSwipeEvents();
    
});

// Navigation entre slides
function showSlide(n) {
    // Masquer toutes les slides
    document.querySelectorAll('.slide-container').forEach(slide => {
        slide.classList.remove('slide-active');
    });
    
    // Afficher la slide demandée
    const slideToShow = document.getElementById(`slide-${n}`);
    if (slideToShow) {
        slideToShow.classList.add('slide-active');
        currentSlide = n;
        
        // Mettre à jour la barre de progression
        updateProgressBar();
        
        // Mettre à jour les numéros de slide
        updateSlideNumbers();
        
        // Arrêter et redémarrer l'autoplay
        restartAutoPlay();
    }
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        showSlide(currentSlide + 1);
    } else {
        // Revenir à la première slide
        showSlide(1);
    }
}

function prevSlide() {
    if (currentSlide > 1) {
        showSlide(currentSlide - 1);
    } else {
        // Aller à la dernière slide
        showSlide(totalSlides);
    }
}

// Gestion des touches du clavier
function handleKeydown(e) {
    switch(e.key) {
        case 'ArrowRight':
        case ' ':
        case 'PageDown':
            nextSlide();
            e.preventDefault();
            break;
            
        case 'ArrowLeft':
        case 'PageUp':
            prevSlide();
            e.preventDefault();
            break;
            
        case 'Home':
            showSlide(1);
            e.preventDefault();
            break;
            
        case 'End':
            showSlide(totalSlides);
            e.preventDefault();
            break;
            
        case 'f':
        case 'F':
            toggleFullscreen();
            e.preventDefault();
            break;
            
        case 'p':
        case 'P':
            printPresentation();
            e.preventDefault();
            break;
    }
}

// Mise à jour de la barre de progression
function updateProgressBar() {
    const progress = ((currentSlide - 1) / (totalSlides - 1)) * 100;
    document.getElementById('progress-bar').style.width = `${progress}%`;
}

// Mise à jour des numéros de slide
function updateSlideNumbers() {
    document.querySelectorAll('.slide-number').forEach((element, index) => {
        element.textContent = `${index + 1}/${totalSlides}`;
    });
}

// Gestion du plein écran
function toggleFullscreen() {
    const elem = document.documentElement;
    
    if (!isFullscreen) {
        if (elem.requestFullscreen) {
            elem.requestFullscreen();
        } else if (elem.webkitRequestFullscreen) { /* Safari */
            elem.webkitRequestFullscreen();
        } else if (elem.msRequestFullscreen) { /* IE11 */
            elem.msRequestFullscreen();
        }
        isFullscreen = true;
        document.getElementById('fullscreen-btn').innerHTML = '<i class="fas fa-compress"></i>';
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) { /* Safari */
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) { /* IE11 */
            document.msExitFullscreen();
        }
        isFullscreen = false;
        document.getElementById('fullscreen-btn').innerHTML = '<i class="fas fa-expand"></i>';
    }
}

// Détection de sortie du mode plein écran
document.addEventListener('fullscreenchange', handleFullscreenChange);
document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
document.addEventListener('msfullscreenchange', handleFullscreenChange);

function handleFullscreenChange() {
    isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
    document.getElementById('fullscreen-btn').innerHTML = isFullscreen ? 
        '<i class="fas fa-compress"></i>' : '<i class="fas fa-expand"></i>';
}

// Impression
function printPresentation() {
    window.print();
}

// Gestes tactiles (swipe)
function setupSwipeEvents() {
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    document.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
}

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0) {
            // Swipe gauche -> slide suivante
            nextSlide();
        } else {
            // Swipe droite -> slide précédente
            prevSlide();
        }
    }
}

// Autoplay
function startAutoPlay() {
    autoPlayInterval = setInterval(nextSlide, autoPlayDelay);
}

function stopAutoPlay() {
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
    }
}

function restartAutoPlay() {
    stopAutoPlay();
    startAutoPlay();
}

// Arrêter l'autoplay lors de l'interaction utilisateur
document.addEventListener('keydown', () => restartAutoPlay());
document.addEventListener('click', () => restartAutoPlay());
document.addEventListener('touchstart', () => restartAutoPlay());

// Navigation par clic sur la barre de progression
document.getElementById('progress-bar').addEventListener('click', function(e) {
    const progressBar = e.target;
    const clickPosition = e.offsetX;
    const progressBarWidth = progressBar.offsetWidth;
    const percentage = clickPosition / progressBarWidth;
    const slideNumber = Math.ceil(percentage * totalSlides);
    showSlide(slideNumber);
});

// Effets visuels supplémentaires
document.querySelectorAll('.objective-card, .data-card, .recommendation-card, .team-member').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.zIndex = '10';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.zIndex = '1';
    });
});

// Initialisation des images de secours
function setupFallbackImages() {
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('error', function() {
            const initials = this.alt.split(' ').map(n => n[0]).join('');
            this.src = `https://via.placeholder.com/150/667eea/ffffff?text=${initials}`;
        });
    });
}

// Exporter les fonctions pour le débogage
window.presentation = {
    nextSlide,
    prevSlide,
    showSlide,
    toggleFullscreen,
    printPresentation,
    currentSlide: () => currentSlide,
    totalSlides: () => totalSlides
};

console.log('Présentation CO₂ Seattle chargée avec succès!');
console.log('Commandes disponibles:');
console.log(' - Flèches gauche/droite: Navigation');
console.log(' - Espace/PageDown: Slide suivante');
console.log(' - PageUp: Slide précédente');
console.log(' - F: Plein écran');
console.log(' - P: Imprimer');
>>>>>>> bc88a2f407130e92528502cbea11695ac8f8f9f7
console.log(' - window.presentation pour API JavaScript');