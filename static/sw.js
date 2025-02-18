self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open("face-recognition-cache").then((cache) => {
            return cache.addAll([
                "/",
                "/templates/Home.html",
                "/static/manifest.json",
                "/static/icons/FaceApp.png"
            ]).catch((error) => {
                console.error("Cache addAll failed: ", error);
            });
        })
    );
});

self.addEventListener("fetch", (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        }).catch((error) => {
            console.error("Fetch failed: ", error);
        })
    );
});
