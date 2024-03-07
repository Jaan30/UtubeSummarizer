document.getElementById('submit').addEventListener('click', LoadVideo);
function extractVideoId(url) {
    var match = url.match(/(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/);
    return match && match[1] ? match[1] : '';
}
urlTextBox.addEventListener("keyup", function (event) {
    if (event.key === "Enter" && urlTextBox.value.trim() !== "") {
        LoadVideo();
    }
});
function LoadVideo() {
    var url = document.getElementById('urlTextBox').value;
    var language = document.getElementById('languageSelector').value;
    if (url.trim() === '') {
        alert('Please enter a valid YouTube video URL.');
        return;
    } 
    var embedUrl = 'https://www.youtube.com/embed/' + extractVideoId(url) + '?cc_lang_pref=' + language;
    document.getElementById('container').innerHTML = '<iframe width="560" height="315" src="' + embedUrl + '" frameborder="0" allowfullscreen></iframe>';
    
}


