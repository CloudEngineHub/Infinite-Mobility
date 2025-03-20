<div align="center">
<img src="logo.png" width="1000"></img>
</div>

# [Infinite Mobility: Scalable High-Fidelity Synthesis of Articulated Objects via Procedural Generation]()
[**Getting Started**](#getting-started)
| [**Contributing**](#contributing)

## What is this?
This repo is about the procedural generation of articulated objects like those in this scene.
<div align="center">
<img src="./gifs/scene.gif" width="1000"></img>
</div>

## TO DO
- [x] Initial release of the code
- [ ] Make installation more self-contained
- [ ] Add usd format support
- [ ] Generate interactive environment with infinigen layout solvers.

## Getting Started

First, follow installation instruction of Infinigen to setup the basic environment [Installation Instructions](docs/Installation.md).  
__important: we are based on an earlier version of Infinigen.(git 572bfe7)  
It could break the pipeline if the newest version is installed__   
Then, please run setup.py to configure our dependencies.
```bash
python setup.py
```
Next, download our part dataset [here](https://github.com/Intern-Nexus/Infinite-Mobility/releases/tag/v0.0.1) 
Finally, configure dataset path in [python code](infinigen/assets/utils/auxiliary_parts.py) line 30.
For example, if part dataset is /datasets/parts
```python
AUXILIARY_PATH = "/datasets/parts"
```
It should work just fine now!🎊 

## Generate
We provide a script for you to generate as many articulated objects as you like!
```bash
python paralled_generate.py <Factory Name> <Number> <MaxProcess>
```
If you want to generate 100 officechairs, and generate 10 samples at the same time, run
```bash
python paralled_generate.py OfficeChairFactory 100 10
```
Results will be in outputs folder in the form of URDF.

For Factory Name we support now, refer to the form below.
| Factory Name |GIFs|
|----|----|
| OfficeChairFactory  | <img src="./gifs/10.gif" width="300"></img> |
| BarChairFactory  |<img src="./gifs/1.gif" width="300"></img> |
| BeverageFridgeFactory  |<img src="./gifs/2.gif" width="300"></img> |    
| DishwasherFactory  |  <img src="./gifs/3.gif" width="300"></img> |
| MicrowaveFactory  |   <img src="./gifs/9.gif" width="300"></img> |
| OvenFactory  | <img src="./gifs/11.gif" width="300"></img> |
| TVFactory|  <img src="./gifs/23.gif" width="300"></img> |
| TapFactory  | <img src="./gifs/18.gif" width="300"></img> |
| ToiletFactory  |  <img src="./gifs/19.gif" width="300"></img> |
| LitedoorFactory |   <img src="./gifs/8.gif" width="300"></img> |
| LampFactory |  <img src="./gifs/7.gif" width="300"></img> |
| PlateOnRackBaseFactory |  <img src="./gifs/13.gif" width="300"></img> |
| KitchenCabinetFactory |   <img src="./gifs/6.gif" width="300"></img> |
| VaseFactory |   <img src="./gifs/21.gif" width="300"></img> |
| BottleFactory |   <img src="./gifs/4.gif" width="300"></img> |
| CocktailTableFactory |  <img src="./gifs/16.gif" width="300"></img> |
| DiningTableFactory| <img src="./gifs/17.gif" width="300"></img> |
| PotFactory |<img src="./gifs/14.gif" width="300"></img> |
| PanFactory |   <img src="./gifs/12.gif" width="300"></img> |
| WindowFactory |  <img src="./gifs/22.gif" width="300"></img> |

## Visualize
```bash
python show.py <path to dir or urdf file>
```
If you want to visualize one urdf file in ./outputs/***.urdf, run
```bash
python show.py ./outputs/***.urdf
```
We also support visualize all urdfs in one dir.If you want to visualize all urdfs in ./output, run
```bash
python show.py ./outputs
```
<div align="center">
<img src="./gifs/chair.gif" width="1000"></img>
</div>

## Contributing

We welcome contributions! You can contribute in many ways:
- **Contribute code to Infinigen repository** - Procedural generators for more categories with interactive parts are still needed, we are happy to update our articulation modifications as Infinigen evolves! 
- **Contribute more diverse parts with fine geometry** - Compared to original infinigen, our work introduces substitution of certain parts with collected meshes. A more diversed parts dataset would boost our performance in many applications! 

## Join Us

We are seeking engineers, interns, researchers, and PhD candidates. If you have an interest in 3D content generation, please send your resume to lvzhaoyang@pjlab.org.cn.

### Getting Help
If you are having difficulties running scripts in our repo, please open issues!
We are happy to provide supports and have discussions!
