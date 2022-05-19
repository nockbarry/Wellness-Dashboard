## Embed YouTube Video in Markdown File

1. Markdown style
```md
[![Watch the video](https://img.youtube.com/vi/nTQUwghvy5Q/default.jpg)](https://youtu.be/nTQUwghvy5Q)
```

2. HTML style
```html
<a href="http://www.youtube.com/watch?feature=player_embedded&v=nTQUwghvy5Q" target="_blank">
 <img src="http://img.youtube.com/vi/nTQUwghvy5Q/mqdefault.jpg" alt="Watch the video" width="240" height="180" border="10" />
</a>
```

3. embed thumbnail
```md
![alt text][image]
[image]: /full/path/to/image/file.jpg "alt_txt"
```

4. resize thumbnail
```md
<img src="/full/path/to/image/file.jpg" alt="alt_text" width="200">
```

### Extras on Thumbnail Format  

Each YouTube video has 4 generated images. The first one in the list is a full size image and others are thumbnail images. They are predictably formatted as follows:

```md
https://img.youtube.com/vi/<insert-youtube-video-id-here>/0.jpg
https://img.youtube.com/vi/<insert-youtube-video-id-here>/1.jpg
https://img.youtube.com/vi/<insert-youtube-video-id-here>/2.jpg
https://img.youtube.com/vi/<insert-youtube-video-id-here>/3.jpg
```

The default thumbnail image (ie. one of 1.jpg, 2.jpg, 3.jpg) is:

```md
https://img.youtube.com/vi/<insert-youtube-video-id-here>/default.jpg  
```

Other than the default.jpg, the following thumbnail formats are also available:

* hqdefault.jpg <- high quality  
* mqdefault.jpg <- medium quality  
* sddefault.jpg <- standard definition  
* maxresdefault.jpg <- maximum resolution  

_NB: sddefault.jpg and maxresdefault.jpg are not always available._

All of the above urls are available over http too. Additionally, the slightly shorter hostname `i3.ytimg.com` works in place of `img.youtube.com` in the example urls above.
