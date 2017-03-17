import { Component, OnInit } from '@angular/core';
import { ClassMetadataService } from './class-metadata.service';

enum HazardLevel {
    HAZARD,
    SECTOR,
    FACILITY,
    COMPONENT
}

@Component({
    selector: 'app-root',
    template: `
    <element-chooser *ngIf="!currentComponent"
        (publish)="doPublish($event)">
    </element-chooser>
    <element-editor *ngIf="currentComponent"
        [value]="currentComponent"
        (publish)="doPublish($event)">
    </element-editor>

    <br/>
    <div *ngIf="levels">
        <table>
        <tr>
        <td>
            <label>Hazard:</label>
        </td>
        <td>
            <ng-select
                [items]="levels.hazards"
                [active]="hazard"
                (selected)="selected($event, hazardLevel.HAZARD)"
                placeholder="please choose a hazard">
            </ng-select>
        </td>
        </tr>
        <tr>
        <td>
            <label>Sector:</label>
        </td>
        <td>
            <ng-select
                [items]="levels.sectors"
                [active]="sector"
                (selected)="selected($event, hazardLevel.SECTOR)"
                placeholder="please choose a sector">
            </ng-select>
        </td>
        </tr>
        <tr>
        <td>
            <label>Facility:</label>
        </td>
        <td>
            <ng-select
                [items]="levels.facilities"
                [active]="facility"
                (selected)="selected($event, hazardLevel.FACILITY)"
                placeholder="please choose a facility type">
            </ng-select>
        </td>
        </tr>
        <tr>
        <td>
            <label>Component:</label>
        </td>
        <td>
            <ng-select
                [items]="levels.components"
                [active]="component"
                (selected)="selected($event, hazardLevel.COMPONENT)"
                placeholder="please choose a component type">
            </ng-select>
        </td>
        </tr>
        <tr>
        <td>
            <label>Name:</label>
        </td>
        <td>
            <input type="text" [value]="name" (change)="setName($event)" style="width: 100%"><br>
        </td>
        </tr>
        <tr>
        <td>
            <label>Description:</label><br>
        </td>
        <td>
            <textarea
                rows="4" cols="50"
                [value]="description"
                (change)="setDescription($event)"></textarea>
        </td>
        </tr>
        </table>

        <button (click)="save()">Save</button>
        <button (click)="reset()">Reset</button>
        <button (click)="resetSave()">Clear type</button>
    </div>
    <br/>

    <div *ngIf="components.length">
        <ul>
            <li *ngFor="let c of components" (click)="showComponent(c.id)">{{c.name}}</li>
        </ul>
    </div>
    `
})
export class AppComponent implements OnInit {
    // unfortunately globals are not visible within templates
    public hazardLevel = HazardLevel;

    // Object containing the result to be saved.
    private resultObject: any = null;
    private levels: any = null;

    // strings that represent the category
    private hazard: string[] = [];
    private sector: string[] = [];
    private facility: string[] = [];
    private component: string[] = [];
    private name: string = null;
    private description: string = null;

    // the previously defined component to display
    private currentComponent: any = null;

    // The existing components (at this stage, response models)
    private components: any = [];

    constructor(private classMetadataService: ClassMetadataService) {}

    ngOnInit() {
        this.getLevels();
        // get the existing components
        this.classMetadataService.getInstancesOf('sifra.elements.ResponseModel')
            .subscribe(
                components => this.components = components,
                error => alert(error)
            );
    }

    doPublish($event) {
        this.resultObject = $event;
    }

    selected($event, level) {
        switch(level) {
            case HazardLevel.HAZARD:
                this.hazard = [$event.id];
                break;
            case HazardLevel.SECTOR:
                this.sector = [$event.id];
                break;
            case HazardLevel.FACILITY:
                this.facility = [$event.id];
                break;
            case HazardLevel.COMPONENT:
                this.component = [$event.id];
                break;
        }
        this.getLevels();
    }

    setName($event) {
        this.name = $event.target.value;
    }

    setDescription($event) {
        this.description = $event.target.value;
    }

    getLevels() {
        this.classMetadataService.getTypeLists(
            this.hazard[0], this.sector[0], this.facility[0], this.component[0]
        ).subscribe(
            levels => this.levels = levels,
            error => alert(error)
        );
    }

    save() {
        if(!this.resultObject) {
            alert('No result to save!');
            return;
        }

        if(
            !this.hazard.length ||
            !this.sector.length ||
            !this.facility.length ||
            !this.component.length
        ) {
            alert('Please select a hazard, sector, facility and component types');
            return;
        }

        if(!this.name || !this.description) {
            alert('Please provide a name and description.');
            return;
        }

        let componentSector = {
            hazard: this.hazard[0],
            sector: this.sector[0],
            facility_type: this.facility[0],
            component: this.component[0]};

        let attrs = {
            name: this.name,
            description: this.description};

        this.resultObject['component_sector'] = componentSector;
        this.resultObject['attributes'] = attrs;

        this.classMetadataService.save(this.resultObject).subscribe(
            newComponent => {
                this.components.push(newComponent);
                delete this.resultObject['component_sector'];
                delete this.resultObject['attributes'];
            },
            error => {
                alert(error);
                delete this.resultObject['component_sector'];
                delete this.resultObject['attributes'];
            });
    }

    reset() {
        this.currentComponent = null;
        this.resultObject = null;
    }

    resetSave() {
        this.component = [];
        this.facility = [];
        this.hazard = [];
        this.sector = [];
        this.name = null;
        this.description = null;
        this.getLevels();
    }

    showComponent(componentId: string) {
        this.classMetadataService.getInstance(componentId)
            .subscribe(
                component => this.currentComponent = component,
                error => alert(error)
            );
    }
}

