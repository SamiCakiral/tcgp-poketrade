// Fonctions de tri communes pour dashboard.html et add_card.html

// Fonction pour convertir une rareté en valeur numérique pour le tri
function rarityValue(rarity) {
    if (!rarity) return 0;
    if (rarity === 'Common') return 1;
    if (rarity.startsWith('Diamond')) {
        return 2 + parseInt(rarity.split(' ')[1] || 0);
    }
    if (rarity.startsWith('Star')) {
        return 6 + parseInt(rarity.split(' ')[1] || 0);
    }
    if (rarity === 'Crown Rare') return 10;
    return 0;
}

// Fonction pour appliquer un tri
function applySortToCards(cards, sortType, sortOrder) {
    let sortedCards;
    
    switch (sortType) {
        case 'collection':
            sortedCards = cards.sort((a, b) => {
                const result = sortOrder === 'asc' 
                    ? a.setId.localeCompare(b.setId) 
                    : b.setId.localeCompare(a.setId);
                return result !== 0 ? result : a.cardNumber - b.cardNumber;
            });
            break;
            
        case 'rarity':
            sortedCards = cards.sort((a, b) => {
                const result = sortOrder === 'asc'
                    ? rarityValue(a.rarity) - rarityValue(b.rarity)
                    : rarityValue(b.rarity) - rarityValue(a.rarity);
                return result !== 0 ? result : (a.setId.localeCompare(b.setId) || a.cardNumber - b.cardNumber);
            });
            break;
            
        case 'name':
            sortedCards = cards.sort((a, b) => {
                return sortOrder === 'asc'
                    ? a.name.localeCompare(b.name)
                    : b.name.localeCompare(a.name);
            });
            break;
            
        case 'number':
            sortedCards = cards.sort((a, b) => {
                return sortOrder === 'asc'
                    ? a.cardNumber - b.cardNumber
                    : b.cardNumber - a.cardNumber;
            });
            break;

        case 'missing':
            // Tri par cartes manquantes
            sortedCards = cards.sort((a, b) => {
                // Si nous avons des attributs owned (logique principale pour dashboard)
                if ('isOwned' in a && 'isOwned' in b) {
                    const aOwned = a.isOwned;
                    const bOwned = b.isOwned;

                    // Mettre les cartes manquantes en premier (non possédées)
                    if (!aOwned && bOwned) return -1;
                    if (aOwned && !bOwned) return 1;
                } else {
                     // Fallback si isOwned n'est pas défini, utiliser les classes DOM
                    const aOwned = a.element && (a.element.getAttribute('data-owned') === 'true' || a.element.classList.contains('owned'));
                    const bOwned = b.element && (b.element.getAttribute('data-owned') === 'true' || b.element.classList.contains('owned'));

                    if (!aOwned && bOwned) return -1;
                    if (aOwned && !bOwned) return 1;
                }

                // Si même statut, trier par set et numéro
                return a.setId.localeCompare(b.setId) || a.cardNumber - b.cardNumber;
            });
            break;

        default:
            sortedCards = cards;
    }
    
    return sortedCards;
}

// Fonction pour créer l'interface utilisateur du tri
function createSortUI(container, onSortApplied) {
    // Créer le bouton flottant
    const sortButton = document.createElement('div');
    sortButton.className = 'sort-floating-button';
    sortButton.innerHTML = `
        <button class="sort-btn" id="open-sort-menu">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="4" y1="9" x2="20" y2="9"></line>
                <line x1="4" y1="15" x2="20" y2="15"></line>
                <line x1="10" y1="3" x2="8" y2="21"></line>
                <line x1="18" y1="4" x2="14" y2="20"></line>
            </svg>
        </button>
    `;
    
    // Créer le menu de tri
    const sortMenu = document.createElement('div');
    sortMenu.className = 'sort-menu';
    sortMenu.id = 'sort-menu';
    sortMenu.innerHTML = `
        <div class="sort-header">
            <h3>Trier les cartes</h3>
            <button class="close-sort-menu">&times;</button>
        </div>
        <div class="sort-options">
            <div class="sort-group">
                <div class="sort-group-title">Par collection</div>
                <div class="sort-buttons">
                    <button class="sort-option" data-sort="collection" data-order="asc">A → Z</button>
                    <button class="sort-option" data-sort="collection" data-order="desc">Z → A</button>
                </div>
            </div>
            
            <div class="sort-group">
                <div class="sort-group-title">Par rareté</div>
                <div class="sort-buttons">
                    <button class="sort-option" data-sort="rarity" data-order="asc">Commune → Rare</button>
                    <button class="sort-option" data-sort="rarity" data-order="desc">Rare → Commune</button>
                </div>
            </div>
            
            <div class="sort-group">
                <div class="sort-group-title">Par nom de carte</div>
                <div class="sort-buttons">
                    <button class="sort-option" data-sort="name" data-order="asc">A → Z</button>
                    <button class="sort-option" data-sort="name" data-order="desc">Z → A</button>
                </div>
            </div>
            
            <div class="sort-group">
                <div class="sort-group-title">Par numéro</div>
                <div class="sort-buttons">
                    <button class="sort-option" data-sort="number" data-order="asc">Croissant</button>
                    <button class="sort-option" data-sort="number" data-order="desc">Décroissant</button>
                </div>
            </div>
            
            <div class="sort-group">
                <div class="sort-group-title">Par cartes manquantes</div>
                <div class="sort-buttons">
                    <button class="sort-option" data-sort="missing" data-group="false">Sans regroupement</button>
                    <button class="sort-option" data-sort="missing" data-group="true">Par collection</button>
                </div>
            </div>
        </div>
    `;
    
    // Ajouter les éléments au conteneur
    container.appendChild(sortButton);
    container.appendChild(sortMenu);
    
    // Gérer les événements
    const sortMenuBtn = sortButton.querySelector('#open-sort-menu');
    const closeSortMenuBtn = sortMenu.querySelector('.close-sort-menu');
    const sortOptions = sortMenu.querySelectorAll('.sort-option');
    
    // Ouvrir/fermer le menu
    sortMenuBtn.addEventListener('click', function() {
        sortMenu.classList.add('open');
    });
    
    closeSortMenuBtn.addEventListener('click', function() {
        sortMenu.classList.remove('open');
    });
    
    // Fermer le menu si on clique en dehors
    window.addEventListener('click', function(event) {
        if (!sortMenu.contains(event.target) && event.target !== sortMenuBtn) {
            sortMenu.classList.remove('open');
        }
    });
    
    // Gérer le choix de tri
    sortOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Mettre à jour la classe active
            sortOptions.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Récupérer les options de tri
            const sortType = this.getAttribute('data-sort');
            const sortOrder = this.getAttribute('data-order');
            const sortGroup = this.getAttribute('data-group');
            
            // Sauvegarder les préférences
            localStorage.setItem('pokeTrade_sortType', sortType);
            localStorage.setItem('pokeTrade_sortOrder', sortOrder);
            if (sortGroup) {
                localStorage.setItem('pokeTrade_sortGroup', sortGroup);
            }
            
            // Fermer le menu
            sortMenu.classList.remove('open');
            
            // Appeler le callback
            if (typeof onSortApplied === 'function') {
                onSortApplied(sortType, sortOrder, sortGroup === 'true');
            }
        });
    });
    
    // Mettre à jour la restauration de l'option active pour inclure data-group
    const savedSortType = localStorage.getItem('pokeTrade_sortType');
    const savedSortOrder = localStorage.getItem('pokeTrade_sortOrder');
    const savedSortGroup = localStorage.getItem('pokeTrade_sortGroup');
    
    if (savedSortType) {
        let selector = `.sort-option[data-sort="${savedSortType}"]`;
        
        if (savedSortOrder) {
            selector += `[data-order="${savedSortOrder}"]`;
        }
        
        if (savedSortGroup && document.querySelector(selector + `[data-group="${savedSortGroup}"]`)) {
            selector += `[data-group="${savedSortGroup}"]`;
        }
        
        const activeOption = sortMenu.querySelector(selector);
        if (activeOption) {
            activeOption.classList.add('active');
        }
    }
    
    return {
        sortType: savedSortType,
        sortOrder: savedSortOrder,
        sortGroup: savedSortGroup
    };
}

// Ajouter ces fonctions de tri globales à la fin du fichier

// Fonction pour aplatir et trier toutes les cartes (sans regroupement)
function flattenAndSortCards(sortType, sortOrder, viewSelector) {
    // Cacher temporairement tous les conteneurs de séries d'origine
    const originalSetContainers = document.querySelectorAll(`${viewSelector} .set-container:not(.flattened-cards-container)`);
    originalSetContainers.forEach(container => {
        container.style.display = 'none';
    });
    
    // Créer ou afficher le conteneur pour toutes les cartes aplaties
    let flatContainer = document.querySelector(`${viewSelector} .flattened-cards-container`);
    if (!flatContainer) {
        flatContainer = document.createElement('div');
        flatContainer.className = 'flattened-cards-container set-container';
        flatContainer.innerHTML = `
            <div class="set-header">
                <h2>Toutes les cartes</h2>
            </div>
            <div class="cards-grid" id="flattened-grid-${viewSelector.replace('.', '')}"></div>
        `;
        document.querySelector(viewSelector).appendChild(flatContainer);
    } else {
        flatContainer.style.display = 'block';
    }
    
    // Collecter toutes les cartes
    const allCards = [];
    originalSetContainers.forEach(container => {
        container.querySelectorAll('.card-box').forEach(cardBox => {
            const clone = cardBox.cloneNode(true);
            console.log(cardBox);
            // Récupérer les données importantes
            const setId = cardBox.getAttribute('data-set-id');
            const cardNumber = parseInt(cardBox.getAttribute('data-card-number'));
            const cardName = cardBox.getAttribute('data-card-name') || '';
            const isOwned = cardBox.classList.contains('owned');
            const isWanted = cardBox.classList.contains('wanted');
            const rarity = cardBox.getAttribute('data-rarity') || '';
            
            allCards.push({
                element: clone,
                setId: setId,
                cardNumber: cardNumber,
                name: cardName,
                isOwned: isOwned,
                isWanted: isWanted,
                rarity: rarity,
                originalCard: cardBox
            });
        });
    });
    
    // Trier les cartes
    const sortedCards = applySortToCards(allCards, sortType, sortOrder);
    
    // Gestion spécifique des cartes manquantes
    if (sortType === 'missing') {
        sortedCards.sort((a, b) => {
            const result = a.isOwned ? 1 : -1;
            return result !== 0 ? result : (a.setId.localeCompare(b.setId) || a.cardNumber - b.cardNumber);
        });
    }
    
    // Afficher les cartes triées
    const grid = document.getElementById(`flattened-grid-${viewSelector.replace('.', '')}`);
    grid.innerHTML = '';
    
    sortedCards.forEach(card => {
        const cardElement = card.element;
        
        // Ajouter des gestionnaires d'événements au clone
        cardElement.addEventListener('click', function() {
            // Trouver la carte originale et simuler un clic dessus
            if (card.originalCard) {
                card.originalCard.click();
            }
        });
        
        grid.appendChild(cardElement);
    });
    
    // Marquer que nous sommes en mode aplati
    localStorage.setItem('pokeTrade_viewMode', 'flattened');
}

// Fonction pour trier les sets et les cartes à l'intérieur
function sortSetsAndCards(sortType, sortOrder, viewSelector) {
    // Masquer le conteneur aplati s'il existe
    const flatContainer = document.querySelector(`${viewSelector} .flattened-cards-container`);
    if (flatContainer) {
        flatContainer.style.display = 'none';
    }
    
    // Afficher tous les conteneurs de séries originaux
    const setContainers = document.querySelectorAll(`${viewSelector} .set-container:not(.flattened-cards-container)`);
    setContainers.forEach(container => {
        container.style.display = 'block';
    });
    
    // Collecter les infos des séries
    const sets = [];
    setContainers.forEach(container => {
        const setHeader = container.querySelector('.set-header h2');
        if (!setHeader) return;
        
        const setId = setHeader.textContent.split(' - ')[0].trim();
        const ownedCards = container.querySelectorAll('.card-box.owned').length;
        const totalCards = container.querySelectorAll('.card-box').length;
        
        sets.push({
            element: container,
            setId: setId,
            ownedCount: ownedCards,
            totalCount: totalCards,
            completionRate: totalCards > 0 ? ownedCards / totalCards : 0
        });
        
        // Trier les cartes à l'intérieur de chaque série
        sortCardsWithinSet(container, sortType, sortOrder);
    });
    
    // Trier les séries
    let sortedSets;
    if (sortType === 'collection') {
        sortedSets = sets.sort((a, b) => {
            return sortOrder === 'asc'
                ? a.setId.localeCompare(b.setId)
                : b.setId.localeCompare(a.setId);
        });
    } else if (sortType === 'missing') {
        sortedSets = sets.sort((a, b) => {
            return a.completionRate - b.completionRate;
        });
    } else {
        // Pour les autres types de tri, on garde l'ordre des séries
        sortedSets = sets;
    }
    
    // Réorganiser les séries dans le DOM
    const container = document.querySelector(viewSelector);
    if (!container) {
        console.error("Conteneur parent non trouvé pour réorganiser les séries:", viewSelector);
        return; // Sortir si le conteneur n'est pas trouvé
    }
    // Vider le conteneur avant d'ajouter les éléments triés (si nécessaire selon la structure)
    // container.innerHTML = ''; // Attention: efface tout, potentiellement problématique si d'autres éléments existent. Préférer append si possible.
    sortedSets.forEach(set => {
        container.appendChild(set.element); // Assure que les éléments sont déplacés/ajoutés dans le bon ordre
    });

    // Marquer que nous sommes en mode regroupé
    localStorage.setItem('pokeTrade_viewMode', 'grouped');
}

// Fonction pour trier les cartes à l'intérieur d'un set
function sortCardsWithinSet(setContainer, sortType, sortOrder) {
    const cardsGrid = setContainer.querySelector('.cards-grid');
    if (!cardsGrid) return;
    
    // Récupérer toutes les cartes du set
    const cards = [];
    cardsGrid.querySelectorAll('.card-box').forEach(cardBox => {
        const setId = cardBox.getAttribute('data-set-id');
        const cardNumber = parseInt(cardBox.getAttribute('data-card-number'));
        const cardName = cardBox.getAttribute('data-card-name') || '';
        const isOwned = cardBox.classList.contains('owned');
        const isWanted = cardBox.classList.contains('wanted');
        const rarity = cardBox.getAttribute('data-rarity') || '';
        
        cards.push({
            element: cardBox,
            setId: setId,
            cardNumber: cardNumber,
            name: cardName,
            isOwned: isOwned,
            isWanted: isWanted,
            rarity: rarity
        });
    });
    
    // Appliquer le tri
    const sortedCards = applySortToCards(cards, sortType, sortOrder);
    
    // Réorganiser les cartes dans le DOM
    const fragment = document.createDocumentFragment();
    sortedCards.forEach(card => {
        fragment.appendChild(card.element);
    });
    
    cardsGrid.innerHTML = '';
    cardsGrid.appendChild(fragment);
} 