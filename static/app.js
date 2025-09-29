class PhotoClusterApp {
    constructor() {
        this.currentPath = '';
        // Корневая папка для обработки очереди
        this.initialPath = '';
        this.queue = [];
        this.lastTasksStr = '';
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadInitialData();
        this.startTaskPolling();
    }

    initializeElements() {
        this.driveButtons = document.getElementById('driveButtons');
        this.currentPathEl = document.getElementById('currentPath');
        this.folderContents = document.getElementById('folderContents');
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.queueList = document.getElementById('queueList');
        this.processBtn = document.getElementById('processBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.includeExcludedBtn = document.getElementById('includeExcludedBtn');
        this.includeExcluded = false;
        this.addQueueBtn = document.getElementById('addQueueBtn');
        this.tasksList = document.getElementById('tasksList');
        this.clearTasksBtn = document.getElementById('clearTasksBtn');
        
        // Проверяем что все элементы найдены
        const elements = {
            driveButtons: this.driveButtons,
            currentPathEl: this.currentPathEl,
            folderContents: this.folderContents,
            uploadZone: this.uploadZone,
            fileInput: this.fileInput,
            queueList: this.queueList,
            processBtn: this.processBtn,
            clearBtn: this.clearBtn,
            addQueueBtn: this.addQueueBtn,
            tasksList: this.tasksList,
            clearTasksBtn: this.clearTasksBtn
        };
        
        for (const [name, element] of Object.entries(elements)) {
            if (!element) {
                console.error(`Element not found: ${name}`);
            }
        }
    }

    setupEventListeners() {
        // Разрешить drop в очередь
        this.queueList.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.queueList.classList.add('drag-over');
        });
        this.queueList.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.queueList.classList.remove('drag-over');
        });
        this.queueList.addEventListener('drop', (e) => {
            e.preventDefault();
            this.queueList.classList.remove('drag-over');
            const path = e.dataTransfer.getData('text/plain');
            if (path) this.addToQueue(path);
        });
        // Кнопки обработки очереди
        this.processBtn.addEventListener('click', () => this.processQueue());
        this.clearBtn.addEventListener('click', () => this.clearQueue());
        this.includeExcludedBtn.addEventListener('click', async () => {
            this.includeExcluded = !this.includeExcluded;
            this.includeExcludedBtn.classList.toggle('active', this.includeExcluded);
            
            console.log(`🔍 Кнопка "Общие" нажата. includeExcluded = ${this.includeExcluded}`);
            
            if (this.includeExcluded) {
                // Ждем добавления папок 'Общие' в очередь
                console.log('🔍 Добавляем исключенные папки в очередь...');
                await this.addExcludedFoldersToQueue();
            }
            // Запускаем обработку очереди с учётом включенных 'общих' папок
            console.log(`🔍 Запускаем processQueue с includeExcluded = ${this.includeExcluded}`);
            await this.processQueue();
        });
        // Кнопка добавить в очередь
        this.addQueueBtn.addEventListener('click', () => this.addToQueue(this.currentPath));
        // Кнопка очистки завершенных задач
        this.clearTasksBtn.addEventListener('click', () => this.clearCompletedTasks());

        // Загрузка файлов
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));

        // Drag & Drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('drag-over');
        });

        this.uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('drag-over');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('drag-over');
            this.handleFileUpload(e.dataTransfer.files);
        });
    }

    async loadInitialData() {
        await this.loadDrives();
        await this.loadQueue();
    }

    async loadDrives() {
        try {
            const response = await fetch('/api/drives');
            const drives = await response.json();
            
            this.driveButtons.innerHTML = '';
            drives.forEach(drive => {
                const button = document.createElement('button');
                button.className = 'drive-btn';
                button.textContent = drive.name;
                button.addEventListener('click', () => this.navigateToFolder(drive.path));
                this.driveButtons.appendChild(button);
            });
        } catch (error) {
            this.showNotification('Ошибка загрузки дисков: ' + error.message, 'error');
        }
    }

    async navigateToFolder(path) {
        try {
            this.currentPath = path;
            // Сохраняем корневую директорию только один раз
            if (!this.initialPath) {
                this.initialPath = path;
            }
            const response = await fetch(`/api/folder?path=${encodeURIComponent(path)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            this.currentPathEl.innerHTML = `<strong>Текущая папка:</strong> ${path}`;
            await this.displayFolderContents(data.contents);
            
        } catch (error) {
            this.showNotification('Ошибка доступа к папке: ' + error.message, 'error');
        }
    }

    async displayFolderContents(contents) {
        this.folderContents.innerHTML = '';
        
        if (contents.length === 0) {
            this.folderContents.innerHTML = `
                <p style="text-align: center; color: #666; padding: 40px 0;">
                    Папка пуста
                </p>
            `;
            return;
        }

        for (const item of contents) {
            // Навигационная кнопка Назад
            if (item.name.includes('⬅️')) {
                const button = document.createElement('button');
                button.className = 'folder-btn back';
                button.setAttribute('draggable', 'true');
                button.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', item.path);
                    e.dataTransfer.effectAllowed = 'move';
                });
                button.textContent = item.name;
                if (item.is_directory) button.addEventListener('click', () => this.navigateToFolder(item.path));
                this.folderContents.appendChild(button);
                continue;
            }
            if (item.is_directory) {
                // Папка: если есть изображения, показываем превью, иначе кнопка
                let imgs = [];
                try {
                    const res = await fetch(`/api/folder?path=${encodeURIComponent(item.path)}`);
                    const folderData = await res.json();
                    imgs = folderData.contents.filter(c => !c.is_directory);
                } catch {}
                if (imgs.length > 0) {
                    // Превью папки
                    const div = document.createElement('div');
                    div.className = 'thumbnail';
                    div.setAttribute('draggable','true');
                    div.addEventListener('click', () => this.navigateToFolder(item.path));
                    
                    // Drag & Drop для папки
                    div.addEventListener('dragstart', e => {
                        e.dataTransfer.setData('text/plain', item.path);
                        e.dataTransfer.effectAllowed = 'move';
                    });
                    div.addEventListener('dragover', e => {
                        e.preventDefault();
                        div.classList.add('drag-over');
                    });
                    div.addEventListener('dragleave', e => {
                        e.preventDefault();
                        div.classList.remove('drag-over');
                    });
                    div.addEventListener('drop', e => {
                        e.preventDefault();
                        div.classList.remove('drag-over');
                        const src = e.dataTransfer.getData('text/plain');
                        this.moveItem(src, item.path);
                    });
                    
                    const img = document.createElement('img');
                    img.src = `/api/image/preview?path=${encodeURIComponent(imgs[0].path)}&size=150`;
                    img.alt = item.name.replace('📂 ', '');
                    div.appendChild(img);
                    
                    // Добавляем подпись с названием папки
                    const caption = document.createElement('div');
                    caption.className = 'thumbnail-caption';
                    caption.textContent = item.name.replace('📂 ', '');
                    div.appendChild(caption);
                    
                    this.folderContents.appendChild(div);
                } else {
                    // Обычная папка без превью
                    const button = document.createElement('button');
                    button.className = 'folder-btn';
                    
                    // Проверяем, является ли папка исключаемой
                    const folderName = item.name.replace('📂 ', '');
                    const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
                    const folderNameLower = folderName.toLowerCase();
                    
                    let isExcluded = false;
                    let excludedName = '';
                    for (const name of excludedNames) {
                        if (folderNameLower.includes(name)) {
                            isExcluded = true;
                            excludedName = name;
                            break;
                        }
                    }
                    
                    if (isExcluded) {
                        button.className += ' disabled';
                        button.textContent = folderName + ' (не обрабатывается)';
                        button.title = `Папки с названием "${excludedName}" не обрабатываются`;
                        button.disabled = true;
                    } else {
                        button.textContent = folderName;
                        button.addEventListener('click', () => this.navigateToFolder(item.path));
                        
                        // Drag & Drop для обычной папки
                        button.setAttribute('draggable', 'true');
                        button.addEventListener('dragstart', e => {
                            e.dataTransfer.setData('text/plain', item.path);
                            e.dataTransfer.effectAllowed = 'move';
                        });
                        button.addEventListener('dragover', e => {
                            e.preventDefault();
                            button.classList.add('drag-over');
                        });
                        button.addEventListener('dragleave', e => {
                            e.preventDefault();
                            button.classList.remove('drag-over');
                        });
                        button.addEventListener('drop', e => {
                            e.preventDefault();
                            button.classList.remove('drag-over');
                            const src = e.dataTransfer.getData('text/plain');
                            this.moveItem(src, item.path);
                        });
                    }
                    
                    this.folderContents.appendChild(button);
                }
                continue;
            }
            // Изображение файла
            if (!item.is_directory && item.name.match(/\.(jpg|jpeg|png|bmp|tif|tiff|webp)$/i)) {
                const div = document.createElement('div');
                div.className = 'thumbnail';
                div.setAttribute('draggable', 'true');
                
                // Drag & Drop для изображения
                div.addEventListener('dragstart', e => {
                    e.dataTransfer.setData('text/plain', item.path);
                    e.dataTransfer.effectAllowed = 'move';
                });
                div.addEventListener('dragover', e => {
                    e.preventDefault();
                    div.classList.add('drag-over');
                });
                div.addEventListener('dragleave', e => {
                    e.preventDefault();
                    div.classList.remove('drag-over');
                });
                div.addEventListener('drop', e => {
                    e.preventDefault();
                    div.classList.remove('drag-over');
                    const src = e.dataTransfer.getData('text/plain');
                    this.moveItem(src, item.path);
                });
                
                const img = document.createElement('img');
                img.src = `/api/image/preview?path=${encodeURIComponent(item.path)}&size=150`;
                img.alt = item.name.replace('🖼 ', '');
                div.appendChild(img);
                
                // Добавляем подпись с названием файла
                const caption = document.createElement('div');
                caption.className = 'thumbnail-caption';
                caption.textContent = item.name.replace('🖼 ', '');
                div.appendChild(caption);
                
                this.folderContents.appendChild(div);
                continue;
            }
            // Другие файлы: просто кнопка
            const button = document.createElement('button');
            button.className = 'folder-btn';
            button.textContent = item.name;
            this.folderContents.appendChild(button);
        }

        // Добавляем кнопку "Добавить в очередь" если это не навигационная кнопка
        if (!contents.some(item => item.name.includes('⬅️'))) {
            const addButton = document.createElement('button');
            addButton.className = 'action-btn';
            addButton.style.marginTop = '15px';
            addButton.textContent = '📌 Добавить в очередь';
            addButton.addEventListener('click', () => this.addToQueue(this.currentPath));
            this.folderContents.appendChild(addButton);
        }
    }

    formatFileSize(bytes) {
        const sizes = ['Б', 'КБ', 'МБ', 'ГБ'];
        if (bytes === 0) return '0 Б';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    async handleFileUpload(files) {
        if (!this.currentPath) {
            this.showNotification('Выберите папку для загрузки файлов', 'error');
            return;
        }

        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`/api/upload?path=${encodeURIComponent(this.currentPath)}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            let successCount = 0;
            let errorCount = 0;
            
            result.results.forEach(item => {
                if (item.status === 'uploaded' || item.status === 'extracted') {
                    successCount++;
                } else {
                    errorCount++;
                }
            });

            if (successCount > 0) {
                this.showNotification(`Загружено файлов: ${successCount}`, 'success');
                // Обновляем содержимое папки
                this.navigateToFolder(this.currentPath);
            }
            
            if (errorCount > 0) {
                this.showNotification(`Ошибок при загрузке: ${errorCount}`, 'error');
            }

        } catch (error) {
            this.showNotification('Ошибка загрузки файлов: ' + error.message, 'error');
        }

        // Очищаем input
        this.fileInput.value = '';
    }

    async addExcludedFoldersToQueue() {
        try {
            // Используем initialPath для поиска общих папок
            const rootPath = this.initialPath || this.currentPath;
            const response = await fetch(`/api/folder?path=${encodeURIComponent(rootPath)}`);
            const data = await response.json();
            
            const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
            const excludedFolders = [];
            
            // Находим все папки с исключаемыми названиями
            for (const item of data.items) {
                if (item.type === 'folder') {
                    const folderName = item.name.replace('📂 ', '');
                    const folderNameLower = folderName.toLowerCase();
                    
                    for (const excludedName of excludedNames) {
                        if (folderNameLower.includes(excludedName)) {
                            excludedFolders.push(item.path);
                            break;
                        }
                    }
                }
            }
            
            // Добавляем найденные папки в очередь с флагом includeExcluded
            for (const folderPath of excludedFolders) {
                await this.addToQueueDirect(folderPath, true);
            }
            
            if (excludedFolders.length > 0) {
                this.showNotification(`Добавлено ${excludedFolders.length} папок "Общие" в очередь`, 'success');
                await this.loadQueue();
            } else {
                this.showNotification('Папки "Общие" не найдены в текущей директории', 'info');
            }
        } catch (error) {
            this.showNotification('Ошибка поиска папок "Общие": ' + error.message, 'error');
        }
    }

    async addToQueueDirect(path, includeExcluded = false) {
        try {
            const url = includeExcluded ? '/api/queue/add?includeExcluded=true' : '/api/queue/add';
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: path })
            });

            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || result.message);
            }
            
            return result;
        } catch (error) {
            this.showNotification('Ошибка добавления в очередь: ' + error.message, 'error');
            throw error;
        }
    }

    async addToQueue(path) {
        // Сохраняем корневую директорию для обработки
        this.initialPath = path;
        // Если не включена обработка исключенных папок, проверяем их
        if (!this.includeExcluded) {
            const excludedNames = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"];
            const pathLower = path.toLowerCase();
            for (const excludedName of excludedNames) {
                if (pathLower.includes(excludedName)) {
                    this.showNotification(`Папки с названием "${excludedName}" не обрабатываются`, 'error');
                    return;
                }
            }
        }
        
        try {
            const result = await this.addToQueueDirect(path);
            this.showNotification(result.message, 'success');
            await this.loadQueue();
        } catch (error) {
            // Ошибка уже обработана в addToQueueDirect
        }
    }

    async loadQueue() {
        try {
            const response = await fetch('/api/queue');
            const data = await response.json();
            this.queue = data.queue;
            this.displayQueue();
        } catch (error) {
            console.error('Ошибка загрузки очереди:', error);
        }
    }

    displayQueue() {
        if (this.queue.length === 0) {
            this.queueList.innerHTML = `
                <p style="text-align: center; color: #666; padding: 20px 0;">
                    Очередь пуста
                </p>
            `;
            this.processBtn.disabled = true;
            this.clearBtn.disabled = true;
            this.addQueueBtn.disabled = false;
        } else {
            this.queueList.innerHTML = '';
            this.queue.forEach((path, index) => {
                const item = document.createElement('div');
                item.className = 'queue-item';
                item.innerHTML = `
                    <span>${index + 1}. ${path}</span>
                `;
                this.queueList.appendChild(item);
            });
            this.processBtn.disabled = false;
            this.clearBtn.disabled = false;
            this.addQueueBtn.disabled = false;
        }
    }

    async processQueue() {
        try {
            this.processBtn.disabled = true;
            this.processBtn.innerHTML = '<div class="loading"></div> Запуск...';

            // Обновляем очередь перед запуском, чтобы избежать гонки состояний
            await this.loadQueue();
            if (!this.queue || this.queue.length === 0) {
                this.showNotification('Очередь пуста. Добавьте папки перед запуском.', 'error');
                return;
            }

            const url = `/api/process?includeExcluded=${this.includeExcluded}`;
            console.log(`🔍 Отправляем запрос: ${url}`);
            const response = await fetch(url, { method: 'POST' });
            const result = await response.json();
            if (!response.ok) {
                // Показать текст ошибки из detail
                this.showNotification(result.detail || result.message || 'Ошибка при запуске обработки', 'error');
                return;
            }
            this.showNotification(result.message, 'success');
            
            await this.loadQueue();
            
        } catch (error) {
            this.showNotification('Ошибка запуска обработки: ' + error.message, 'error');
        } finally {
            this.processBtn.disabled = false;
            this.processBtn.innerHTML = '🚀 Обработать очередь';
        }
    }

    async clearQueue() {
        try {
            const response = await fetch('/api/queue', {
                method: 'DELETE'
            });

            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue();

        } catch (error) {
            this.showNotification('Ошибка очистки очереди: ' + error.message, 'error');
        }
    }

    async loadTasks() {
        try {
            const response = await fetch('/api/tasks');
            const data = await response.json();
            
            // Обновляем только если есть изменения
            const newTasksStr = JSON.stringify(data.tasks);
            if (this.lastTasksStr !== newTasksStr) {
                this.lastTasksStr = newTasksStr;
                this.displayTasks(data.tasks);
            }
            
        } catch (error) {
            console.error('Ошибка загрузки задач:', error);
        }
    }

    displayTasks(tasks) {
        if (!this.tasksList) {
            console.error('tasksList element not found!');
            return;
        }
        
        // Фильтруем только активные задачи для отображения
        const activeTasks = tasks.filter(task => 
            task.status === 'running' || task.status === 'pending'
        );
        
        if (activeTasks.length === 0) {
            this.tasksList.innerHTML = `
                <p style="text-align: center; color: #666; padding: 40px 0;">
                    Активных задач нет
                </p>
            `;
            return;
        }

        this.tasksList.innerHTML = '';
        
        // Сортируем только активные задачи
        activeTasks.sort((a, b) => {
            const order = { 'running': 0, 'pending': 1 };
            return order[a.status] - order[b.status];
        });

        activeTasks.forEach(task => {
            const taskEl = document.createElement('div');
            taskEl.className = `task-item ${task.status}`;
            
            const statusEmoji = {
                'pending': '⏳',
                'running': '⚡',
                'completed': '✅',
                'error': '❌'
            };

            let resultHtml = '';
            if (task.status === 'completed' && task.result) {
                resultHtml = `
                    <div class="result-stats">
                        <div class="stat-item">
                            <div class="stat-value moved">${task.result.moved}</div>
                            <div class="stat-label">Перемещено</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value copied">${task.result.copied}</div>
                            <div class="stat-label">Скопировано</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value clusters">${task.result.clusters_count}</div>
                            <div class="stat-label">Кластеров</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value no-faces">${task.result.no_faces_count}</div>
                            <div class="stat-label">Без лиц</div>
                        </div>
                    </div>
                `;
            }

            let progressHtml = '';
            if (task.status === 'running' || task.status === 'pending') {
                const progress = task.progress || 0;
                progressHtml = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div class="progress-text">${progress}%</div>
                    <div class="progress-details">${task.message || 'Подготовка...'}</div>
                `;
            }

            taskEl.innerHTML = `
                <div class="task-header">
                    <span>${statusEmoji[task.status]} ${task.path}</span>
                    <button class="task-close" data-task-id="${task.id}">×</button>
                </div>
                ${progressHtml}
                ${resultHtml}
            `;
            this.tasksList.appendChild(taskEl);
        });
    }

    async startTaskPolling() {
        setInterval(async () => {
            await this.loadTasks();
        }, 1000); // Проверять каждую секунду
    }

    async clearCompletedTasks() {
        try {
            const response = await fetch('/api/tasks/clear', {
                method: 'DELETE'
            });
            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadTasks();
        } catch (error) {
            this.showNotification('Ошибка очистки завершенных задач: ' + error.message, 'error');
        }
    }

    async moveItem(srcPath, destPath) {
        try {
            const response = await fetch(`/api/move?srcPath=${encodeURIComponent(srcPath)}&destPath=${encodeURIComponent(destPath)}`, {
                method: 'POST'
            });
            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue(); // Обновляем очередь после перемещения
        } catch (error) {
            this.showNotification('Ошибка перемещения файла: ' + error.message, 'error');
        }
    }

    async deleteItem(path) {
        if (!confirm('Вы уверены, что хотите удалить этот файл/папку?')) {
            return;
        }
        try {
            const response = await fetch(`/api/delete?path=${encodeURIComponent(path)}`, {
                method: 'DELETE'
            });
            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue(); // Обновляем очередь после удаления
        } catch (error) {
            this.showNotification('Ошибка удаления файла/папки: ' + error.message, 'error');
        }
    }

    async showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    async loadQueue() {
        try {
            const response = await fetch('/api/queue');
            const data = await response.json();
            this.queue = data.queue;
            this.displayQueue();
        } catch (error) {
            console.error('Ошибка загрузки очереди:', error);
        }
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    const app = new PhotoClusterApp();
});