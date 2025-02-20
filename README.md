<div align="center">
<img src="logo.png" width="1000"></img>
</div>

# [Infinite Mobility: Scaling High-fidelity Articulated Object Synthesize using Procedural Generation]()
[**Getting Started**](#getting-started)
| [**Papers**](#papers)
| [**Contributing**](#contributing)

## What is this?
This repo is about the procedural generation of articulated objects like this.
<div align="center">
<img src="./gifs/cabinet.gif" width="500"></img>
</div>

## Getting Started

First, follow installation instruction of Infinigen to setup the basic emnvironment [Installation Instructions](docs/Installation.md).  
__important: we are based on an earlier version of Infinigen.  
It could break the pipeline if the newest version is installed__   
Then, just run setup.py to configure our dependency.  
Finally, download our part dataset [here](https://github.com/yinoqifu00/Infinite-Mobility/releases/tag/v0.0.1) and configure dataset path in [python code](infinigen/assets/utils/auxiliary_parts.py).  
It should work just fine now!🎊 

## Generate
We provide a script for you to generate as many articulated objects as you like!
```bash
python paralled_generate.py <Factory Name> <Number> <MaxProcess>
```
Results will be in outputs folder in the form of URDF.

For Factory Name we support now, refer to the form below.
| Factory Name |
|----|
| OfficeChairFactory  |
| BarChairFactory  |
| BeverageFridgeFactory  |    
| DishwasherFactory  |  
| MicrowaveFactory  |   
| OvenFactory  | 
| TVFactory|  
| TapFactory  | 
| ToiletFactory  |  
| HardwareFactory |  
| LitedoorFactory |   
| LampFactory |  
| PlateOnRackBaseFactory |  
| KitchenCabinetFactory |   
| VaseFactory |   
| BottleFactory |   
| CocktailTableFactory |  
| DiningTableFactory| 
| PotFactory |
| PanFactory |   
| LidFactory |
| WindowFactory |  

## Visualize
```bash
python show.py <path to your urdf or dir of your urdfs>
```
<div align="center">
<img src="./gifs/chair.gif" width="1000"></img>
</div>

## Papers

If you use Infinite Mobility in your work, please cite following academic papers:

<h3 align="center"><a href="">Infinite Mobility: Scaling High-fidelity Articulated Object Synthesize using Procedural Generation</a></h3>
<p align="center">
Coming soon!
</p>

<h3 align="center"><a href="https://arxiv.org/pdf/2306.09310">Infinite Photorealistic Worlds using Procedural Generation</a></h3>
<p align="center">
<a href="http://araistrick.com/">Alexander Raistrick</a>*, 
<a href="https://www.lahavlipson.com/">Lahav Lipson</a>*, 
<a href="https://mazeyu.github.io/">Zeyu Ma</a>* (*equal contribution, alphabetical order) <br>
<a href="https://www.cs.princeton.edu/~lm5483/">Lingjie Mei</a>, 
<a href="https://www.cs.princeton.edu/~mingzhew">Mingzhe Wang</a>, 
<a href="https://zuoym15.github.io/">Yiming Zuo</a>, 
<a href="https://kkayan.com/">Karhan Kayan</a>, 
<a href="https://hermera.github.io/">Hongyu Wen</a>, 
<a href="https://pvl.cs.princeton.edu/people.html">Beining Han</a>, <br>
<a href="https://pvl.cs.princeton.edu/people.html">Yihan Wang</a>, 
<a href="http://www-personal.umich.edu/~alnewell/index.html">Alejandro Newell</a>, 
<a href="https://heilaw.github.io/">Hei Law</a>, 
<a href="https://imankgoyal.github.io/">Ankit Goyal</a>, 
<a href="https://yangky11.github.io/">Kaiyu Yang</a>, 
<a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
Conference on Computer Vision and Pattern Recognition (CVPR) 2023
</p>

```
@inproceedings{infinigen2023infinite,
  title={Infinite Photorealistic Worlds Using Procedural Generation},
  author={Raistrick, Alexander and Lipson, Lahav and Ma, Zeyu and Mei, Lingjie and Wang, Mingzhe and Zuo, Yiming and Kayan, Karhan and Wen, Hongyu and Han, Beining and Wang, Yihan and Newell, Alejandro and Law, Hei and Goyal, Ankit and Yang, Kaiyu and Deng, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12630--12641},
  year={2023}
}
```

<h3 align="center"><a href="https://arxiv.org/abs/2406.11824">Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation</a></h3>
<p align="center">
<a href="http://araistrick.com/">Alexander Raistrick</a>*, 
<a href="https://www.cs.princeton.edu/~lm5483/">Lingjie Mei</a>*, 
<a href="https://kkayan.com/">Karhan Kayan</a>*, (*equal contribution, random order) <br>
<a href="https://david-yan1.github.io/">David Yan</a>, 
<a href="https://zuoym15.github.io/">Yiming Zuo</a>, 
<a href="https://pvl.cs.princeton.edu/people.html">Beining Han</a>, 
<a href="https://hermera.github.io/">Hongyu Wen</a>, 
<a href="https://scholar.google.com/citations?user=q38OfTQAAAAJ&hl=en">Meenal Parakh</a>, <br>
<a href="https://stamatisalex.github.io/">Stamatis Alexandropoulos</a>, 
<a href="https://www.lahavlipson.com/">Lahav Lipson</a>, 
<a href="https://mazeyu.github.io/">Zeyu Ma</a>,
<a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
Conference on Computer Vision and Pattern Recognition (CVPR) 2024
</p>

```
@inproceedings{infinigen2024indoors,
    author    = {Raistrick, Alexander and Mei, Lingjie and Kayan, Karhan and Yan, David and Zuo, Yiming and Han, Beining and Wen, Hongyu and Parakh, Meenal and Alexandropoulos, Stamatis and Lipson, Lahav and Ma, Zeyu and Deng, Jia},
    title     = {Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {21783-21794}
}
```

## Contributing

We welcome contributions! You can contribute in many ways:
- **Contribute code to Infinigen repository** - Procedural generators for more categories with interactive parts are still needed, we are happy to update our articulation modifications as Infinigen evolves! 
- **Contribute more diverse parts with fine geometry** - Compared to original infinigen, our work introduces substitution of certain parts with collected meshes. A more diversed parts dataset would boost our performance in many applications! 

### Getting Help
If you are having difficulties running scripts in our repo, please open issues!
We are happy to provide supports and have discussions!
