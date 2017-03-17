import { Component, Input, Output, OnInit, ChangeDetectorRef, EventEmitter } from '@angular/core';
import { SelectComponent } from 'ng2-select';
import { ClassMetadataService } from '../class-metadata.service';

@Component({
    selector: 'element-chooser',
    template: `
    <div>
        <div *ngIf="name">
            {{name}}
        </div>
        <div>
            <ng-select
                [items]="classTypes"
                (selected)="selected($event);"
                placeholder="please choose an element type to create">
            </ng-select>
            <element-editor *ngIf="className"
                [name]="null"
                [className]="className"
                (publish)="doPublish($event)">
            </element-editor>
            <button (click)="reset();">Reset</button>
        </div>
    </div>
    `
})
export class ElementChooserComponent implements OnInit {
    @Input() name: string = null;

    @Output() publish = new EventEmitter();

    classTypes: any = null;
    className: string = null;
    dirty: boolean = false;

    constructor(
        private changeDetector: ChangeDetectorRef,
        private classMetadataService: ClassMetadataService
    ) {}

    ngOnInit() {
        this.classMetadataService.getClassTypes().subscribe(
            cs => this.classTypes = cs,
            error => alert(<any>error)
        );
    }

    selected($event) {
        if(this.dirty) {
            alert('please save or discard current changes');
            return;
        }
        this.reset();
        this.changeDetector.detectChanges();
        this.className = $event.id;
    }

    reset() {
        this.className = '';
        this.dirty = false;
    }

    doPublish($event) {
        if(this.name) {
            this.publish.emit({name:this.name, value: $event});
        } else {
            this.publish.emit($event);
        }
        this.dirty = true;
    }
}

